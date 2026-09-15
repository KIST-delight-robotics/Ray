"""reSpeaker 마이크 → GPT-Live → 스피커. 오디오 외의 모든 서버 이벤트를 순서대로 출력.

공식 예제 audio_transcript.py에서 WAV 입력을 마이크 스트리밍으로 바꾼 것.
    uv run python scripts/gpt_live/mic_live.py [--seconds 60] [--no-play]

- 세션 오디오 포맷은 16 kHz(reSpeaker 캡처 레이트와 동일 → 리샘플 없음). 입·출력 공유 포맷.
- 스피커 소리가 마이크로 되돌아가면(에코 캔슬 없음) 모델이 자기 말을 듣게 되니 볼륨을 낮추거나 --no-play로 시작.
- --seconds 가 지나면 session.close 를 보내고 session.closed 를 기다린 뒤 종료. Ctrl+C 도 동일.
"""

from __future__ import annotations

import array
import asyncio
import base64
import argparse
import json
import threading
import queue
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import hashlib
import os

import numpy as np
import pyaudio
from openai import AsyncOpenAI

RATE = 16000  # 세션 오디오 포맷. reSpeaker 캡처 레이트와 같게
CAPTURE_CH = 6  # reSpeaker XVF3800 은 6ch 로 열어야 함. ch0 = 처리된 mono
CHUNK = RATE * 20 // 1000  # 20 ms = 320 샘플


def find_input_device(pa: pyaudio.PyAudio, name: str) -> int:
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if name in info["name"].lower() and info["maxInputChannels"] >= CAPTURE_CH:
            return i
    raise SystemExit(f"입력 장치를 찾지 못함: {name}")


def mic_thread(pa: pyaudio.PyAudio, dev: int, loop: asyncio.AbstractEventLoop, out: asyncio.Queue[bytes], stop: threading.Event) -> None:
    stream = pa.open(format=pyaudio.paInt16, channels=CAPTURE_CH, rate=RATE, input=True,
                     input_device_index=dev, frames_per_buffer=CHUNK)
    try:
        while not stop.is_set():
            raw = stream.read(CHUNK, exception_on_overflow=False)
            mono = array.array("h", raw)[0::CAPTURE_CH].tobytes()
            loop.call_soon_threadsafe(out.put_nowait, mono)
    finally:
        stream.stop_stream()
        stream.close()


def speaker_thread(pa: pyaudio.PyAudio, pcm: queue.Queue[bytes | None], prebuffer_ms: int, stats: dict[str, int]) -> None:
    """큐의 오디오를 스피커로 쓴다. prebuffer_ms 만큼 쌓인 뒤에 재생을 시작하고,
    재생 중 다음 조각을 꺼내려는 순간 큐가 비어 있으면(언더런) 다시 prebuffer_ms 만큼 쌓일 때까지 기다린다.
    prebuffer_ms=0 이면 오는 즉시 쓰되, 언더런 횟수는 똑같이 센다."""
    try:
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True)
    except Exception as exc:  # daemon 스레드라 예외가 묻히므로 직접 출력
        print(f"!! 스피커 열기 실패: {exc}")
        return
    need_bytes = RATE * 2 * prebuffer_ms // 1000
    pending: list[bytes] = []
    got = 0
    playing = False
    try:
        while True:
            if playing and pcm.empty():
                stats["underruns"] += 1
                playing = False  # 다시 프리버퍼 채우기로
            chunk = pcm.get()  # 블로킹
            if chunk is None:
                for b in pending:
                    stream.write(b)
                return
            if playing:
                stream.write(chunk)
                continue
            pending.append(chunk)
            got += len(chunk)
            if got >= need_bytes:
                for b in pending:
                    stream.write(b)
                pending, got, playing = [], 0, True
    except Exception as exc:
        print(f"!! 스피커 재생 중 예외: {exc}")
    finally:
        stream.stop_stream()
        stream.close()


# --- 클라이언트 위임 데모 -------------------------------------------------------
# 프롬프팅 가이드 권장 구조: 역할 → 언제 위임하는가 → 언제 위임하지 않는가.
DELEGATION_DEMO_INSTRUCTIONS = """\
You are Ray, a small robot. Speak Korean, briefly and casually.

Delegation policy:
- Backend tools: weather lookup.
- Delegate to the backend when: the user asks about the weather or temperature anywhere.
- Do not delegate to the backend when: the user is chatting, greeting, or asking something you can answer yourself.
While waiting for the backend, say briefly that you are checking; do not guess the result.
"""


# --- Responses 위임(OpenAI 관리 백엔드) ---------------------------------------------
# Live 모델에는 "무엇을 넘길지"만, 백엔드 모델에는 "어떻게 처리할지"를 따로 준다.
RESPONSES_LIVE_INSTRUCTIONS = """\
You are Ray, a small, friendly desk robot.
Speak Korean unless the user asks to switch. Keep a casual, warm tone and short answers.
If the user is frustrated, acknowledge it briefly and focus on the next helpful step.

Backchannel policy: Use moderate backchannels. Acknowledge naturally without competing with the main response.

Interruption policy: Stop speaking when the user interrupts. Listen to what they say.

Delegation policy:
Backend tools:
- Web search: current date and time, weather, news, and facts you are not sure about.
- Robot status: Ray's battery level.
- End of conversation: closes the session when the user is done talking.
Delegate to the backend when:
- The request needs current information or a fact you are not sure about.
- The user asks about Ray's battery.
- The user says goodbye or wants to end the conversation. Say a short goodbye yourself at the same time.
Do not delegate to the backend when:
- You can answer from the conversation or a still-current result.
- The user is chatting, greeting, or thinking aloud.
Delegate before giving an answer that depends on backend work.
Do not guess the result while waiting. Do not mention the backend.
"""
RESPONSES_BACKEND_INSTRUCTIONS = """\
You are the backend for Ray, a Korean-speaking desk robot.

Tools:
- Use web search for current information.
- Call get_battery_level when the user asks about Ray's battery.
- Call end_conversation when the user says goodbye or wants to stop, then reply with an empty message.

Answer in Korean, in one or two short sentences that sound natural when spoken aloud.
No lists, no URLs, no markdown.
"""

# 백엔드가 실제로 어떤 입력을 받는지 보기 위한 탐침용 지시문. 백엔드 출력은 response.event 로 우리에게 보인다.
PROBE_BACKEND_INSTRUCTIONS = """\
You are the backend for a Korean-speaking voice robot named Ray.
Always begin your reply with "요청 확인:" followed by a word-for-word quote of EVERY user and assistant
message you received in this request, in order, each prefixed with its role (user:/assistant:).
Then answer the latest request in one short Korean sentence.
"""

# 백엔드 모델이 부를 수 있는 우리 함수. 실행은 우리 코드(아래 run_function)에서 한다.
RESPONSES_FUNCTION_TOOLS = [
    {
        "type": "function",
        "name": "get_battery_level",
        "description": "Return Ray's current battery level in percent. Call this whenever the user asks about Ray's battery or charge.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function",
        "name": "end_conversation",
        "description": "End the current conversation session. Call this when the user says goodbye or clearly wants to stop talking.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
]


def run_function(name: str, arguments: str) -> str:
    """백엔드가 요청한 함수를 실행해 결과 문자열(JSON)을 돌려준다. 실제 레이에선 여기서 하드웨어/기억 시스템을 부른다."""
    if name == "get_battery_level":
        return json.dumps({"percent": 73, "charging": False})
    if name == "end_conversation":
        print("   *** end_conversation 호출됨 — 실제 레이에서는 여기서 세션 종료 절차 시작 ***")
        return json.dumps({"status": "ending"})
    return json.dumps({"error": f"unknown function {name}"})


def fake_backend(request_text: str) -> str:
    """위임된 요청에 대한 가짜 백엔드. 실제로는 여기서 LLM/툴을 부른다."""
    if "날씨" in request_text or "기온" in request_text:
        return "서울은 지금 맑고 24도야."
    return f"'{request_text.strip()}'에 대해서는 지금 답을 찾지 못했어."


# --- TTS 스크립트 입력(마이크 대신) -------------------------------------------------
TTS_CACHE_DIR = Path(__file__).parent / "tts_cache"
END_CONTINUE = os.environ.get("GPT_LIVE_END_CONTINUE") == "1"  # 실험용: 종료 툴 뒤에도 response.create 를 보낼지


async def synthesize_16k(client: AsyncOpenAI, text: str) -> bytes:
    """OpenAI TTS 로 한국어 문장을 합성해 16 kHz mono PCM16 으로 돌려준다. 같은 문장은 파일 캐시."""
    TTS_CACHE_DIR.mkdir(exist_ok=True)
    cache = TTS_CACHE_DIR / (hashlib.sha1(text.encode()).hexdigest() + ".pcm16k")
    if cache.exists():
        return cache.read_bytes()
    resp = await client.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=text, response_format="pcm")
    pcm24 = np.frombuffer(resp.content, dtype=np.int16).astype(np.float32)  # TTS pcm 은 24 kHz mono
    n_out = int(len(pcm24) * RATE / 24000)
    pcm16 = np.interp(np.linspace(0, 1, n_out, endpoint=False), np.linspace(0, 1, len(pcm24), endpoint=False), pcm24).astype(np.int16).tobytes()
    cache.write_bytes(pcm16)
    return pcm16


def load_script(path: str) -> list[tuple[str, str]]:
    """'say 문장' / 'wait 초' 줄을 읽는다. '#' 으로 시작하는 줄과 빈 줄은 무시."""
    steps = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cmd, _, arg = line.partition(" ")
        if cmd not in ("say", "wait", "instructions", "thinking", "commentary"):
            raise SystemExit(f"스크립트 문법 오류: {line!r}")
        steps.append((cmd, arg))
    return steps


def rms(pcm: bytes) -> int:
    samples = array.array("h", pcm)
    return int((sum(v * v for v in samples) / max(len(samples), 1)) ** 0.5)


async def run(*, model: str, instructions: str, seconds: float, play: bool, save: str | None, prebuffer_ms: int,
              delegation: str, backend_model: str, probe_backend_input: bool, seeds: list[str],
              script: str | None) -> None:
    loop = asyncio.get_running_loop()
    pa = pyaudio.PyAudio()
    mic_q: asyncio.Queue[bytes] = asyncio.Queue()
    spk_q: queue.Queue[bytes | None] = queue.Queue()
    stop_mic = threading.Event()
    t0 = time.monotonic()

    def ts() -> str:
        return f"{time.monotonic() - t0:6.2f}s"

    async with AsyncOpenAI() as client, client.live.connect(max_retries=0) as connection:
        # 1) 세션 시작. 모델·오디오 포맷·instructions 는 여기서 고정된다.
        session_cfg: dict = {
            "model": model,
            "audio": {"format": {"type": "audio/pcm", "rate": RATE}},
            "instructions": instructions,
        }
        if delegation == "responses":
            session_cfg["delegation"] = {
                "type": "responses",
                "responses": {
                    "model": backend_model,
                    "instructions": PROBE_BACKEND_INSTRUCTIONS if probe_backend_input else RESPONSES_BACKEND_INSTRUCTIONS,
                    "tools": [{"type": "web_search"}, *RESPONSES_FUNCTION_TOOLS],
                },
            }
        if seeds:
            # 시작 전 텍스트 히스토리("role:text" 형식). 말하지 않고도 모델이 반응/위임하는지 보기 위한 용도
            items = []
            for seed in seeds:
                role, _, text = seed.partition(":")
                part_type = "output_text" if role == "assistant" else "input_text"
                items.append({"type": "message", "role": role, "content": [{"type": part_type, "text": text}]})
            session_cfg["input"] = items
        await connection.session.start(session=session_cfg)
        # 2) session.started 가 올 때까지 오디오를 보내지 않는다.
        while True:
            event = await asyncio.wait_for(connection.recv(), timeout=30)
            print(f"{ts()} <- {event.type}")
            if event.type == "error":
                raise SystemExit(f"세션 시작 실패: {event.error}")
            if event.type == "session.started":
                remain = event.session.expires_at - time.time()
                print(f"{ts()}    session.id={event.session.id} model={event.session.model} 만료까지 {remain / 60:.1f}분")
                if event.session.delegation is not None:
                    print(f"{ts()}    서버가 확인한 delegation 설정: {event.session.delegation.model_dump_json(exclude_none=True)[:600]}")
                break

        if script is None:
            threading.Thread(target=mic_thread, args=(pa, find_input_device(pa, "respeaker"), loop, mic_q, stop_mic), daemon=True).start()
        else:
            steps = load_script(script)
            # 합성은 미리 전부 해두고(네트워크 지연이 타임라인에 섞이지 않게), 피더는 20 ms 간격으로 큐에 넣는다
            speech = {text: await synthesize_16k(client, text) for cmd, text in steps if cmd == "say"}
            print(f"{ts()} [스크립트] {len(steps)}단계, 합성 {len(speech)}문장 준비 완료")

            async def script_feed() -> None:
                chunk_bytes = CHUNK * 2
                silence = bytes(chunk_bytes)
                period = CHUNK / RATE

                async def stream(pcm: bytes) -> None:
                    for i in range(0, len(pcm), chunk_bytes):
                        mic_q.put_nowait(pcm[i:i + chunk_bytes].ljust(chunk_bytes, b"\0"))
                        await asyncio.sleep(period)

                for cmd, arg in steps:
                    if cmd == "say":
                        print(f"{ts()} [스크립트] say: {arg}")
                        await stream(speech[arg])
                    elif cmd == "wait":
                        await stream(silence * int(float(arg) / period))
                    else:
                        # 세션 중 텍스트 주입 3종. delegation_id=None 은 세션 전체 대상
                        print(f"{ts()} -> session.{cmd}.append content={arg!r}")
                        resource = getattr(connection.session, cmd)
                        await resource.append(content=arg, delegation_id=None, event_id=f"script_{cmd}_{int(time.monotonic() * 1000)}")
                print(f"{ts()} [스크립트] 끝 — 세션 종료")
                script_done.set()

            script_done = asyncio.Event()
            feeder = asyncio.create_task(script_feed())
        spk_thread: threading.Thread | None = None
        spk_stats = {"underruns": 0}
        if play:
            spk_thread = threading.Thread(target=speaker_thread, args=(pa, spk_q, prebuffer_ms, spk_stats), daemon=True)
            spk_thread.start()
        print(f"{ts()} 마이크 스트리밍 시작 — 말하세요 ({seconds:.0f}초 후 자동 종료)")

        # 3) 마이크 → session.input_audio.append (ack 없음)
        async def send_mic() -> None:
            while True:
                chunk = await mic_q.get()
                await connection.session.input_audio.append(audio=base64.b64encode(chunk).decode("ascii"))

        # 4) 서버 이벤트 읽기. 오디오 delta 는 스피커로, 나머지는 전부 출력.
        audio_deltas = 0
        received_pcm: list[bytes] = []  # --save 용
        # 전송 상태 지표 (1초 벽시계 창마다 집계)
        win_start = time.monotonic()
        win_deltas = 0
        win_audio_ms = 0.0
        win_max_gap_ms = 0.0
        win_rms_max = 0
        last_delta_at: float | None = None
        all_gaps_ms: list[float] = []  # 세션 전체 조각 간 간격 (종료 시 히스토그램)
        user_fragments: list[tuple[int, int, str]] = []  # (start_ms, end_ms, text) — 위임 요청 복원용
        pending_calls: list[tuple[str, str, str]] = []  # responses 위임에서 백엔드가 요청한 (call_id, name, arguments)
        handled_until_ms = 0  # 이전 위임에서 이미 사용한 전사 구간의 끝

        async def handle_client_delegation(delegation_id: str, offset_ms: int) -> None:
            """위임 이벤트에는 요청 내용이 없다. offset_ms 이전의 사용자 전사를 이어 붙여 요청을 복원한다."""
            nonlocal handled_until_ms
            await asyncio.sleep(0.3)  # 마지막 전사 조각이 위임 이벤트보다 늦게 올 수 있어 잠깐 기다림
            # offset 직전 발화 하나만: 뒤에서부터 거슬러 가며 조각 사이가 1.5초 넘게 비면 그 앞은 이전 발화로 본다
            cand = [f for f in user_fragments if handled_until_ms <= f[0] <= offset_ms + 1000]
            pieces: list[str] = []
            for i in range(len(cand) - 1, -1, -1):
                st, en, t = cand[i]
                if pieces and cand[i + 1][0] - en > 1500:
                    break
                pieces.insert(0, t)
            request = "".join(pieces)
            handled_until_ms = offset_ms
            print(f"{ts()}    [위임] 복원한 요청: {request!r}")
            await asyncio.sleep(1.5)  # 백엔드가 일하는 시간 흉내
            answer = fake_backend(request)
            print(f"{ts()} -> session.commentary.append delegation_id={delegation_id} content={answer!r}")
            await connection.session.commentary.append(
                content=answer, delegation_id=delegation_id, event_id=f"deleg_{offset_ms}"
            )

        def flush_window(now: float) -> None:
            nonlocal win_start, win_deltas, win_audio_ms, win_max_gap_ms, win_rms_max
            if win_deltas:
                print(f"{ts()}    [전송] 1초 동안 {win_deltas}개 조각 = 오디오 {win_audio_ms:.0f}ms, "
                      f"조각 간 최대 간격 {win_max_gap_ms:.0f}ms, 출력 rms 최대 {win_rms_max}, 재생 대기 큐 {spk_q.qsize()}개, 언더런 누적 {spk_stats['underruns']}")
            win_start, win_deltas, win_audio_ms, win_max_gap_ms, win_rms_max = now, 0, 0.0, 0.0, 0

        last_voice_at: float | None = None  # 마지막으로 0이 아닌 출력 조각을 받은 시각
        END_SILENCE_SEC = 1.0  # 작별 인사가 끝났다고 볼 무음 길이
        END_MAX_WAIT_SEC = 8.0  # 무음이 안 와도 이 시간 뒤엔 닫는다

        async def end_sequence() -> None:
            """레이의 FAREWELL 진입에 해당: 입력을 닫고, 모델의 마지막 말이 끝나길 기다린 뒤 세션을 닫는다."""
            t_start = time.monotonic()
            print(f"{ts()} -> session.input_audio.mute (종료 결정, 더 듣지 않음)")
            await connection.session.input_audio.mute(event_id="end_mute")
            while True:
                await asyncio.sleep(0.1)
                quiet_for = time.monotonic() - (last_voice_at or t_start)
                if quiet_for >= END_SILENCE_SEC:
                    print(f"{ts()}    출력 무음 {quiet_for:.1f}s 지속 → 작별 인사 끝으로 판정")
                    break
                if time.monotonic() - t_start >= END_MAX_WAIT_SEC:
                    print(f"{ts()}    상한 {END_MAX_WAIT_SEC}s 도달 → 강제 종료")
                    break
            print(f"{ts()} -> session.close")
            await connection.session.close(event_id="end_close")

        async def receive() -> None:
            nonlocal audio_deltas, win_deltas, last_voice_at, win_audio_ms, win_max_gap_ms, win_rms_max, last_delta_at
            async for event in connection:
                t = event.type
                now = time.monotonic()
                if now - win_start >= 1.0:
                    flush_window(now)
                if t == "session.output_audio.delta":
                    audio_deltas += 1
                    pcm = base64.b64decode(event.delta)
                    if play:
                        spk_q.put(pcm)
                    if save:
                        received_pcm.append(pcm)
                    win_deltas += 1
                    win_audio_ms += len(pcm) / (RATE * 2) * 1000
                    r = rms(pcm)
                    win_rms_max = max(win_rms_max, r)
                    if r > 30:
                        last_voice_at = now
                    if last_delta_at is not None:
                        gap = (now - last_delta_at) * 1000
                        win_max_gap_ms = max(win_max_gap_ms, gap)
                        all_gaps_ms.append(gap)
                    last_delta_at = now
                    if audio_deltas == 1:
                        print(f"{ts()} <- {t} 첫 조각 {len(pcm)}B = {len(pcm) / (RATE * 2) * 1000:.0f}ms rms={rms(pcm)}")
                elif t == "session.input_transcript.delta":
                    print(f"{ts()} <- USER  [{event.start_ms}-{event.end_ms}ms] {event.delta!r}")
                    user_fragments.append((event.start_ms, event.end_ms, event.delta))
                elif t == "session.output_transcript.delta":
                    print(f"{ts()} <- RAY   [{event.start_ms}-{event.end_ms}ms] {event.delta!r}")
                elif t == "session.delegation.created":
                    print(f"{ts()} <- {t} id={event.delegation.id} target={event.delegation.target} offset={event.offset_ms}ms")
                    if delegation == "client-demo" and event.delegation.target == "client":
                        asyncio.create_task(handle_client_delegation(event.delegation.id, event.offset_ms))
                elif t == "response.event":
                    # Responses 위임: 백엔드(Responses API)의 스트리밍 이벤트가 그대로 중계된다
                    inner = event.event
                    it = inner.get("type")
                    if it == "response.output_text.delta":
                        print(f"{ts()} <- [백엔드] text.delta {inner.get('delta')!r}")
                    elif it == "response.output_item.done":
                        item = inner.get("item", {})
                        if item.get("type") == "function_call":
                            # 함수 호출은 output_item.done 에서만 call_id/name/arguments 가 모두 갖춰진다
                            print(f"{ts()} <- [백엔드] function_call name={item.get('name')} args={item.get('arguments')} call_id={item.get('call_id')}")
                            pending_calls.append((item["call_id"], item["name"], item.get("arguments") or "{}"))
                        else:
                            print(f"{ts()} <- [백엔드] output_item.done type={item.get('type')}")
                    elif it in ("response.created", "response.completed", "response.failed", "response.incomplete"):
                        print(f"{ts()} <- [백엔드] {it} delegation_id={event.delegation_id}")
                        if it == "response.completed" and pending_calls:
                            # 모아둔 함수 호출을 전부 실행해 결과를 넣고, 명시적으로 응답을 이어가게 한다
                            ending = False
                            for call_id, name, args in pending_calls:
                                output = run_function(name, args)
                                ending |= name == "end_conversation"
                                print(f"{ts()} -> response.item.create function_call_output call_id={call_id} output={output}")
                                await connection.response.item.create(
                                    item={"type": "function_call_output", "call_id": call_id, "output": output},
                                    event_id=f"tool_{call_id}",
                                )
                            pending_calls.clear()
                            if ending and not END_CONTINUE:
                                # 종료 툴이면 백엔드 응답을 이어가지 않는다 — 이어가면 백엔드가 작별 인사를 또 만들고
                                # 대화 모델이 그걸 한 번 더 말할 수 있다. 세션은 곧 닫히므로 위임을 미완으로 둔다.
                                print(f"{ts()}    (end_conversation → response.create 생략)")
                                asyncio.create_task(end_sequence())
                            else:
                                print(f"{ts()} -> response.create (함수 결과 반영해 계속)")
                                await connection.response.create(event_id="continue_after_tools")
                                if ending:
                                    asyncio.create_task(end_sequence())
                    elif it.startswith("response.web_search_call"):
                        print(f"{ts()} <- [백엔드] {it}")
                    # 그 외(output_text.done, content_part 등)는 생략
                elif t in ("session.commentary.appended", "session.instructions.appended", "session.thinking.appended"):
                    print(f"{ts()} <- {t} client_event_id={event.client_event_id} [{event.start_ms}-{event.end_ms}ms]")
                elif t == "session.usage.updated":
                    cw = getattr(event, "context_window", None)
                    ratio = f" context_window.usage_ratio={cw.usage_ratio:.4f} (≈{cw.usage_ratio * 128000:,.0f} tok)" if cw is not None else ""
                    print(f"{ts()} <- {t} seconds={event.usage.seconds}{ratio}")
                elif t == "error":
                    print(f"{ts()} <- error {event.error}")
                elif t == "session.closed":
                    print(f"{ts()} <- {t} reason={event.reason} usage.seconds={event.usage.seconds}")
                    return
                else:
                    print(f"{ts()} <- {t}")

        sender = asyncio.create_task(send_mic())
        receiver = asyncio.create_task(receive())
        try:
            waiters = {receiver}
            if script is not None:
                waiters.add(asyncio.create_task(script_done.wait()))  # 스크립트가 끝나면 --seconds 전이라도 종료
            await asyncio.wait(waiters, timeout=seconds, return_when=asyncio.FIRST_COMPLETED)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            stop_mic.set()
            sender.cancel()
            if script is not None:
                feeder.cancel()
            if not receiver.done():
                # 5) 정상 종료: session.close → session.closed 를 기다린다.
                print(f"{ts()} -> session.close")
                await connection.session.close()
                try:
                    await asyncio.wait_for(receiver, timeout=10)
                except asyncio.TimeoutError:
                    print(f"{ts()} session.closed 를 10초 안에 받지 못함")
            spk_q.put(None)
            if spk_thread is not None:
                spk_thread.join(timeout=5)  # 큐에 남은 오디오를 다 쓴 뒤에 PyAudio 를 닫는다
                print(f"재생 언더런(큐 비어서 프리버퍼 재대기) 총 {spk_stats['underruns']}회, 프리버퍼 {prebuffer_ms}ms")
            pa.terminate()
            if all_gaps_ms:
                n = len(all_gaps_ms)
                print(f"[지연 분포] 조각 {n}개, 간격 최대 {max(all_gaps_ms):.0f}ms "
                      f"(정상 100ms; 100 을 넘는 만큼이 늦은 시간)")
                edges = [0, 110, 130, 150, 200, 250, 300, 400, 500, 10_000]
                for lo, hi in zip(edges, edges[1:]):
                    c = sum(1 for g in all_gaps_ms if lo <= g < hi)
                    if c:
                        label = f"{lo}-{hi}ms" if hi < 10_000 else f"{lo}ms+"
                        print(f"  {label:>10}: {c:5d} ({c / n * 100:5.1f}%)")
            if save and received_pcm:
                with wave.open(save, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(RATE)
                    w.writeframes(b"".join(received_pcm))
                print(f"받은 오디오 저장: {save} ({sum(map(len, received_pcm)) / (RATE * 2):.1f}s)")


class Tee:
    """stdout 을 화면과 로그 파일에 동시에 쓴다 (실행마다 scripts/gpt_live/logs/<시각>.log)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", buffering=1)
        self._stdout = sys.stdout

    def write(self, data: str) -> None:
        self._stdout.write(data)
        self._file.write(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gpt-live-1")
    parser.add_argument("--instructions", default="Respond briefly and naturally to the user.")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--no-play", action="store_true", help="모델 오디오를 스피커로 재생하지 않음")
    parser.add_argument("--save", metavar="WAV", help="서버에서 받은 오디오를 그대로 WAV로 저장")
    parser.add_argument("--prebuffer-ms", type=int, default=0, help="재생 시작 전 미리 쌓을 오디오 양(ms). 0이면 오는 즉시 재생")
    parser.add_argument("--delegation", choices=["none", "client-demo", "responses"], default="none",
                        help="none: 위임 이벤트를 찍기만 함 / client-demo: 가짜 백엔드로 답을 돌려줌 / responses: OpenAI 관리 백엔드(web_search)")
    parser.add_argument("--backend-model", default="gpt-5.4-mini", help="responses 위임에 쓸 백엔드 모델")
    parser.add_argument("--probe-backend-input", action="store_true",
                        help="responses 위임에서 백엔드가 받은 입력을 그대로 인용하게 해 문맥 전달 방식을 관찰")
    parser.add_argument("--script", metavar="FILE", help="마이크 대신 스크립트(say/wait 줄)를 TTS 로 합성해 입력으로 흘려 넣음")
    parser.add_argument("--seed", action="append", default=[], metavar="ROLE:TEXT",
                        help="세션 시작 시 input 으로 심을 메시지. 여러 번 지정 가능 (예: --seed 'user:안녕' --seed 'assistant:응, 안녕')")
    args = parser.parse_args()
    if args.instructions == parser.get_default("instructions"):
        if args.delegation == "client-demo":
            args.instructions = DELEGATION_DEMO_INSTRUCTIONS
        elif args.delegation == "responses":
            args.instructions = RESPONSES_LIVE_INSTRUCTIONS
    log_path = Path(__file__).parent / "logs" / f"{datetime.now():%Y%m%d_%H%M%S}.log"
    sys.stdout = Tee(log_path)
    print(f"로그 파일: {log_path}")
    print(f"옵션: {vars(args)}")
    try:
        asyncio.run(run(model=args.model, instructions=args.instructions, seconds=args.seconds,
                        play=not args.no_play, save=args.save, prebuffer_ms=args.prebuffer_ms,
                        delegation=args.delegation, backend_model=args.backend_model,
                        probe_backend_input=args.probe_backend_input, seeds=args.seed, script=args.script))
    except KeyboardInterrupt:
        pass
