#!/usr/bin/env python3
"""장치 없이 Responses 위임 + 대본 thinking.append를 검증한다.

PYTHONPATH=. .venv/bin/python scripts/gpt_live/story_append_probe.py --output /tmp/ray_story_append/run1
출력: 입력/출력 WAV, 전사, 원시 제어 이벤트, 전송 타임라인과 지연 측정. 자동 점수는 참고용이다.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import time
import wave
from pathlib import Path

import numpy as np
import openai
import tiktoken

from voice_pipeline.adapters.gpt_live import (
    GPTLiveSession,
    LiveAudio,
    LiveClosed,
    LiveError,
    LiveFunctionCall,
    LiveResponseDone,
    LiveSessionConfig,
    LiveTranscript,
)
from voice_pipeline.engines.gpt_live.instructions import (
    DEFAULT_LIVE_INSTRUCTIONS,
    LIVE_VOICE,
    build_backend_instructions,
)
from voice_pipeline.engines.gpt_live.story_append import StoryAppendTransfer, narration_text

RATE = 24000
FRAME = RATE // 50 * 2


def synthesize(text: str, cache: Path) -> bytes:
    path = cache / (hashlib.sha256(text.encode()).hexdigest() + ".pcm")
    if not path.exists():
        with openai.OpenAI() as client:
            result = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=text,
                response_format="pcm",
            )
            path.write_bytes(result.content)
    return path.read_bytes()


def tool(name: str, description: str) -> dict:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    }


def run(args: argparse.Namespace) -> None:
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    cache = out.parent / "tts_cache"
    cache.mkdir(exist_ok=True)
    request = synthesize("이야기 하나 들려줘.", cache)
    followups: list[tuple[str, bytes]] = []
    if args.scenario == "backchannel":
        followups = [("응, 그렇구나.", synthesize("응, 그렇구나.", cache))]
    elif args.scenario == "question":
        followups = [
            (text, synthesize(text, cache)) for text in ("잠깐, 벤틀리가 어디에서 태어났다고?", "응, 계속 들려줘.")
        ]
    text = narration_text(args.script.read_text())
    (out / "narration.txt").write_text(text)
    encoding = tiktoken.get_encoding("o200k_base")

    def count(value: str) -> int:
        return len(encoding.encode(value))

    raw = (out / "events.jsonl").open("w")
    start = time.monotonic()

    class RecordedSession(GPTLiveSession):
        def _dispatch(self, event: object) -> None:
            if getattr(event, "type", "") != "session.output_audio.delta":
                raw.write(
                    json.dumps(
                        {"time": time.monotonic() - start, "event": event.model_dump(mode="json")}, ensure_ascii=False
                    )
                    + "\n"
                )
                raw.flush()
            super()._dispatch(event)

    config = LiveSessionConfig(
        instructions=DEFAULT_LIVE_INSTRUCTIONS,
        backend_model="gpt-5.4-mini",
        backend_instructions=build_backend_instructions(),
        tools=(
            tool("start_story", "Select and prepare the available story; application supplies the full script."),
            tool("finish_story", "End the story when requested or when the song offer is declined."),
            tool("play_song", "Play the story's suggested song only when the user explicitly agrees."),
        ),
        voice=LIVE_VOICE,
        sample_rate=RATE,
    )
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(config), ensure_ascii=False, indent=2))
    live = RecordedSession(config)
    transfer = None
    calls: dict[str, LiveFunctionCall] = {}
    ready_call = None
    last_voice = 0.0
    first_story_voice = None
    first_voice_after_signal = None
    start_signal_at = None
    tool_received = None
    early_voice = []
    transcripts = {}
    extra_tools = []
    scheduled = []
    failure = None
    outcome = "duration_limit"
    try:
        live.start()
        start = time.monotonic()
        next_frame = start
        audio_pending = bytes(RATE * 2) + request
        request_finished = False
        followup_index = 0
        last_followup = 0.0
        with wave.open(str(out / "output.wav"), "wb") as output, wave.open(str(out / "input.wav"), "wb") as input_wav:
            for wav in (output, input_wav):
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(RATE)
            while time.monotonic() - start < args.duration:
                now = time.monotonic() - start
                if time.monotonic() >= next_frame:
                    pcm, audio_pending = audio_pending[:FRAME], audio_pending[FRAME:]
                    pcm = pcm.ljust(FRAME, b"\0")
                    live.send_audio(pcm)
                    input_wav.writeframesraw(pcm)
                    next_frame += 0.02
                    if time.monotonic() - next_frame > 0.1:
                        next_frame = time.monotonic() + 0.02
                    request_finished = not audio_pending
                event = live.poll_event()
                if isinstance(event, LiveAudio):
                    output.writeframesraw(event.pcm)
                    samples = np.frombuffer(event.pcm, dtype=np.int16).astype(np.float64)
                    voiced = len(samples) and np.sqrt(np.mean(samples**2)) >= 80
                    if voiced:
                        last_voice = now
                        if transfer and transfer.state in ("preparing", "injecting", "backend"):
                            early_voice.append(now)
                        if start_signal_at is not None and first_voice_after_signal is None:
                            first_voice_after_signal = now
                elif isinstance(event, LiveTranscript):
                    transcripts[event.segment_id] = dataclasses.asdict(event)
                    if event.closed:
                        print(f"[{now:.1f}] {event.speaker}: {event.text}", flush=True)
                    if event.speaker == "assistant" and start_signal_at is not None and first_story_voice is None:
                        if "눈송이" in event.text:
                            first_story_voice = event.start_ms / 1000
                elif isinstance(event, LiveFunctionCall):
                    calls[event.call_id] = event
                    if event.name == "start_story" and transfer is None:
                        tool_received = now
                    else:
                        extra_tools.append({"time": now, **dataclasses.asdict(event)})
                elif isinstance(event, LiveResponseDone):
                    if event.status != "completed":
                        raise RuntimeError(f"backend response {event.status}")
                    for call_id in event.function_call_ids:
                        call = calls.pop(call_id)
                        if call.name == "start_story" and transfer is None:
                            ready_call = call
                        else:
                            # 실제 장치 동작은 수행하지 않으며 성공으로 보고하지 않는다.
                            live.submit_function_output(
                                call_id, json.dumps({"error": "device action disabled in probe"})
                            )
                            live.continue_response()
                elif isinstance(event, LiveError):
                    raise RuntimeError(f"{event.code}: {event.message}")
                elif isinstance(event, LiveClosed):
                    raise RuntimeError(f"session closed: {event.reason}")
                if ready_call is not None and now - last_voice >= 0.5 and request_finished:
                    transfer = StoryAppendTransfer(
                        live, ready_call, args.script.stem, text, count, start_kind=args.start_kind
                    )
                    (out / "parts.json").write_text(json.dumps(transfer.parts, ensure_ascii=False, indent=2))
                    (out / "preparation.txt").write_text(transfer.preparation)
                    transfer.begin(now)
                    ready_call = None
                if transfer:
                    if event is not None:
                        transfer.handle(event, now)
                    transfer.tick(now)
                    signals = [
                        x
                        for x in transfer.timeline
                        if x["event"] == "append_sent"
                        and x["kind"] == args.start_kind
                        and transfer.state in ("starting", "started")
                    ]
                    if signals and start_signal_at is None:
                        start_signal_at = signals[-1]["time"]
                        print(f"[{now:.1f}] start signal; {len(transfer.parts)} script parts accepted", flush=True)
                    if transfer.failure:
                        raise RuntimeError(transfer.failure)
                if first_voice_after_signal is not None and followup_index < len(followups) and not audio_pending:
                    due = first_voice_after_signal + 45 if followup_index == 0 else last_followup + 25
                    if now >= due:
                        words, audio_pending = followups[followup_index]
                        scheduled.append({"time": now, "text": words})
                        last_followup = now
                        followup_index += 1
                assistant = " ".join(t["text"] for t in transcripts.values() if t["speaker"] == "assistant")
                if "들어보실래요" in assistant and now - last_voice > 6 and start_signal_at is not None:
                    outcome = "song_offer_observed"
                    break
                if transfer is None and now > 45:
                    raise RuntimeError("start_story was not called within 45 seconds")
                if event is None:
                    time.sleep(0.003)
    except Exception as exc:
        failure = str(exc)
        outcome = "failed"
        logging.exception("probe failed")
    finally:
        live.close()
        raw.close()
        (out / "transcripts.json").write_text(json.dumps(list(transcripts.values()), ensure_ascii=False, indent=2))
        (out / "transcript.txt").write_text("\n".join(f"{v['speaker']}: {v['text']}" for v in transcripts.values()))
        summary = {
            "session_id": live.session_id,
            "outcome": outcome,
            "failure": failure,
            "scenario": args.scenario,
            "start_kind": args.start_kind,
            "tool_received": tool_received,
            "start_signal": start_signal_at,
            "first_voice_after_signal": first_voice_after_signal,
            "first_story_transcript_start": first_story_voice,
            "early_voice_frames": len(early_voice),
            "extra_tools": extra_tools,
            "followups": scheduled,
            "transfer_state": transfer.state if transfer else None,
            "timeline": transfer.timeline if transfer else [],
            "part_tokens": [count(part) for part in transfer.parts] if transfer else [],
        }
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k != "timeline"}, ensure_ascii=False), flush=True)
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--script", type=Path, default=Path("scripts/gpt_live/fixtures/snowflower_append.md"))
    parser.add_argument("--start-kind", choices=("commentary", "instructions"), default="commentary")
    parser.add_argument("--scenario", choices=("uninterrupted", "backchannel", "question"), default="uninterrupted")
    parser.add_argument("--duration", type=float, default=240)
    run(parser.parse_args())
