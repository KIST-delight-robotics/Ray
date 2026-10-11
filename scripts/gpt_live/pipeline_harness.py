"""실작동 경로 실험 틀 — wiring 으로 조립한 LiveSessionLoop 를 합성 음성 입력·C++ 스텁으로 돌리고 이벤트를 기록한다.

``exp_*.py`` 가 쓴다. 마이크 대신 ``audio_queue`` 에 TTS 음성을 넣고, C++ 대신 :class:`StubBridge` 가 재생 명령을
받는다(모델은 자기 소리를 듣지 않는다). 지시문·루프·백엔드는 실작동 그대로다. 녹음은 왼쪽 사용자(넣은 음성), 오른쪽
모델(받은 출력, 루프가 채우거나 버리기 전)의 스테레오 WAV 다 (:func:`save_conversation_wav`).

기록은 어댑터·루프의 비공개 메서드를 감싸서 한다(:func:`install`). 코드 구조가 바뀌면 여기를 같이 고쳐야 한다.
결과는 ``var/experiments/<실험>/<번호>_<태그>/`` 에 쓴다. 장기기억은 기본으로 끄고(``run_session(memory_enabled=)``),
DB·기기 설정 파일은 실험 폴더의 별도 파일을 쓴다.

하드웨어 대역: 볼륨은 기록만 하고 실제 싱크에 적용하지 않는다(조립 시 복원 적용과 볼륨 툴이 개발 PC·로봇의 볼륨을
바꾸지 않게). 노래 카탈로그는 음원 파일(별도 배포) 없이도 채운다 — C++ 스텁은 음원을 재생하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from exp_audio import resample, write_conversation_wav
from exp_dirs import result_dir

from voice_pipeline import wiring
from voice_pipeline.adapters import gpt_live
from voice_pipeline.adapters.cpp_bridge import CppEvent, CppEventType
from voice_pipeline.adapters.led import LEDController
from voice_pipeline.device_settings import DeviceSettings
from voice_pipeline.engines.gpt_live import songs
from voice_pipeline.engines.gpt_live.loop import LiveSessionLoop
from voice_pipeline.settings import BRIDGE_SAMPLE_RATE, FRAME_SIZE_BYTES, SAMPLE_RATE

REPO = Path(__file__).resolve().parents[2]
TTS_CACHE = Path(__file__).resolve().parent / "tts_cache"  # mic_live.py 와 같은 캐시
FRAME_SEC = FRAME_SIZE_BYTES / (SAMPLE_RATE * 2)
QUIET_RMS = 30  # LiveSessionLoop._QUIET_RMS 와 같게

logger = logging.getLogger("pipeline_harness")


# ---------------------------------------------------------------------------
# 기록
# ---------------------------------------------------------------------------


class Recorder:
    """세션 하나의 기록. 시각은 모두 ``time.monotonic()``.

    Attributes:
        events: ``{"t", "name", ...}`` 목록. delegation_created, backend_call(fn), backend_reply(text),
            append_thinking / append_instructions(text), submit_function_output, dj_begin, song_play(song) 등.
        transcripts: 전사 세그먼트 id → 최신 스냅샷(speaker, text, 첫 도착 t). 닫히지 않은 세그먼트도 남는다.
        audio: 모델 출력 조각마다 (도착 t, 바이트 수, 음성 여부).
        pcm: 받은 모델 출력 오디오 전부 (``BRIDGE_SAMPLE_RATE``). 루프가 채우거나 버리기 전이라 ``audio`` 와
            위치가 맞는다.
        frame_hook: 프레임마다 루프 스레드에서 불린다 ``(loop, now)``. 앱 신호를 흉내 낼 때 쓴다.
        inputs: 첫 음성 뒤에 이어 넣을 사용자 음성(16 kHz PCM). 넣은 순서대로, 앞 음성이 끝나면 흘린다.
        user_clips: 실제로 흘리기 시작한 사용자 음성 (시작 t, 16 kHz PCM). 녹음의 왼쪽 채널.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.events: list[dict[str, Any]] = []
        self.transcripts: dict[str, dict[str, Any]] = {}
        self.audio: list[tuple[float, int, bool]] = []
        self.pcm = bytearray()
        self.connected_at: float | None = None
        self.frame_hook: Callable[[LiveSessionLoop, float], None] | None = None
        self.inputs: queue.Queue[bytes] = queue.Queue()
        self.user_clips: list[tuple[float, bytes]] = []

    def mark(self, name: str, **fields: Any) -> None:
        self.events.append({"t": time.monotonic(), "name": name, **fields})

    def first(self, name: str, *, after: float = 0.0) -> dict[str, Any] | None:
        return next((e for e in self.events if e["name"] == name and e["t"] >= after), None)

    def all(self, name: str, *, after: float = 0.0) -> list[dict[str, Any]]:
        return [e for e in self.events if e["name"] == name and e["t"] >= after]

    def said(self, speaker: str = "assistant") -> list[str]:
        segs = sorted(self.transcripts.values(), key=lambda s: s["t"])
        return [s["text"] for s in segs if s["speaker"] == speaker]

    def voice_segments(self, *, merge_gap: float = 0.35) -> list[tuple[float, float]]:
        """모델 음성 구간 [(시작, 끝)] — 조각 도착 시각 기준, merge_gap 이하 공백은 잇는다."""
        segs: list[list[float]] = []
        for t, n, voiced in self.audio:
            if not voiced:
                continue
            if segs and t - segs[-1][1] <= merge_gap:
                segs[-1][1] = t
            else:
                segs.append([t - n / (BRIDGE_SAMPLE_RATE * 2), t])
        return [(a, b) for a, b in segs]


CURRENT: Recorder | None = None
_installed = False


def _rms(pcm: bytes) -> float:
    s = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(s * s))) if len(s) else 0.0


def install() -> None:
    """어댑터·루프에 기록 훅을 건다 (한 번만). 동작은 바꾸지 않는다."""
    global _installed
    if _installed:
        return
    _installed = True

    orig_dispatch = gpt_live.GPTLiveSession._dispatch

    def dispatch(self: gpt_live.GPTLiveSession, event: Any) -> None:
        rec = CURRENT
        t = getattr(event, "type", None)
        if rec is not None and t == "session.delegation.created":
            rec.mark("delegation_created")
        elif rec is not None and t == "response.event":
            inner = event.event
            if inner.get("type") == "response.output_item.done":
                item = inner.get("item") or {}
                if item.get("type") == "function_call":
                    rec.mark("backend_call", fn=item.get("name"))
                elif item.get("type") == "web_search_call":
                    rec.mark("backend_call", fn="web_search")
                elif item.get("type") == "message":
                    parts = item.get("content") or []
                    text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))
                    rec.mark("backend_reply", text=text.strip())
        elif rec is not None and t in ("session.thinking.appended", "session.instructions.appended"):
            rec.mark(t.split(".")[1] + "_ack")
        orig_dispatch(self, event)

    gpt_live.GPTLiveSession._dispatch = dispatch  # type: ignore[method-assign]

    for name in ("append_thinking", "append_instructions", "append_commentary", "submit_function_output"):
        orig = getattr(gpt_live.GPTLiveSession, name)

        def wrapper(self: Any, *args: Any, _orig: Any = orig, _name: str = name, **kwargs: Any) -> Any:
            if CURRENT is not None:
                CURRENT.mark(_name, text=args[0] if args and isinstance(args[0], str) else None)
            return _orig(self, *args, **kwargs)

        setattr(gpt_live.GPTLiveSession, name, wrapper)

    orig_start = LiveSessionLoop._start_session
    orig_on_audio = LiveSessionLoop._on_audio
    orig_on_transcript = LiveSessionLoop._on_transcript
    orig_dj_begin = LiveSessionLoop._dj_begin
    orig_nudge = LiveSessionLoop._maybe_nudge_dj

    def start_session(self: LiveSessionLoop) -> None:
        orig_start(self)
        if CURRENT is not None:
            CURRENT.connected_at = time.monotonic()

    def on_audio(self: LiveSessionLoop, pcm: bytes) -> None:
        if CURRENT is not None:
            CURRENT.audio.append((time.monotonic(), len(pcm), any(pcm) and _rms(pcm) >= QUIET_RMS))
            CURRENT.pcm.extend(pcm)
        orig_on_audio(self, pcm)

    def on_transcript(self: LiveSessionLoop, seg: gpt_live.LiveTranscript) -> None:
        if CURRENT is not None and seg.text.strip():
            prev = CURRENT.transcripts.get(seg.segment_id)
            CURRENT.transcripts[seg.segment_id] = {
                "t": prev["t"] if prev else time.monotonic(),
                "speaker": seg.speaker,
                "text": seg.text.strip(),
            }
        orig_on_transcript(self, seg)

    def dj_begin(self: LiveSessionLoop) -> None:
        if CURRENT is not None:
            CURRENT.mark("dj_begin")
        orig_dj_begin(self)

    def frame(self: LiveSessionLoop, now: float) -> None:  # _maybe_nudge_dj 는 매 프레임 불린다
        if CURRENT is not None and CURRENT.frame_hook is not None:
            CURRENT.frame_hook(self, now)
        orig_nudge(self, now)

    LiveSessionLoop._start_session = start_session  # type: ignore[method-assign]
    LiveSessionLoop._on_audio = on_audio  # type: ignore[method-assign]
    LiveSessionLoop._on_transcript = on_transcript  # type: ignore[method-assign]
    LiveSessionLoop._dj_begin = dj_begin  # type: ignore[method-assign]
    LiveSessionLoop._maybe_nudge_dj = frame  # type: ignore[method-assign]

    # 하드웨어 대역 — wiring 이 조립 때 읽는 이름을 바꾼다
    songs._SONG_FILES = tuple(f for f in songs._SONG_FILES if f[0] != "audio/music")
    wiring.DeviceSettings = _RecordedVolumeSettings  # type: ignore[misc]


def _record_volume(percent: int) -> None:
    if CURRENT is not None:
        CURRENT.mark("volume_set", percent=percent)
    logger.info("volume set to %d%% (stub, not applied)", percent)


class _RecordedVolumeSettings(DeviceSettings):
    """볼륨을 실제 싱크에 적용하지 않고 기록만 하는 :class:`DeviceSettings`."""

    def __init__(self, path: str | Path, *, led: LEDController) -> None:
        super().__init__(path, led=led, set_volume=_record_volume)


# ---------------------------------------------------------------------------
# 입출력 대역
# ---------------------------------------------------------------------------


class StubBridge:
    """C++ 대신. audio_end·stop 에 바로, 노래에는 ``song_sec`` 뒤 playback_complete 를 준다."""

    def __init__(self, song_sec: float = 2.0) -> None:
        self._events: queue.Queue[CppEvent] = queue.Queue()
        self._song_sec = song_sec

    def send_stream_start(self, *, live: bool = False) -> None:
        pass

    def send_audio(self, pcm: bytes) -> None:
        pass

    def send_audio_end(self) -> None:
        self._events.put(CppEvent(CppEventType.PLAYBACK_COMPLETE))

    def send_stop(self) -> None:
        self._events.put(CppEvent(CppEventType.PLAYBACK_COMPLETE))

    def send_play_audio_csv(self, audio_name: str) -> None:
        if CURRENT is not None:
            CURRENT.mark("song_play", song=audio_name)
        timer = threading.Timer(self._song_sec, lambda: self._events.put(CppEvent(CppEventType.PLAYBACK_COMPLETE)))
        timer.daemon = True
        timer.start()

    def poll_event(self) -> CppEvent | None:
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None


def tts(text: str) -> bytes:
    """OpenAI TTS 로 합성한 16 kHz mono PCM16. 같은 문장은 파일 캐시."""
    TTS_CACHE.mkdir(exist_ok=True)
    cache = TTS_CACHE / (hashlib.sha1(text.encode()).hexdigest() + ".pcm16k")
    if cache.exists():
        return cache.read_bytes()
    import openai

    resp = openai.OpenAI().audio.speech.create(
        model="gpt-4o-mini-tts", voice="alloy", input=text, response_format="pcm"
    )
    pcm24 = np.frombuffer(resp.content, dtype=np.int16).astype(np.float32)  # 24 kHz
    n_out = int(len(pcm24) * SAMPLE_RATE / 24000)
    x_out = np.linspace(0, 1, n_out, endpoint=False)
    pcm16 = np.interp(x_out, np.linspace(0, 1, len(pcm24), endpoint=False), pcm24).astype(np.int16).tobytes()
    cache.write_bytes(pcm16)
    return pcm16


def _feed(q: queue.Queue[bytes], speech: bytes, rec: Recorder, stop: threading.Event) -> None:
    """연결되면 1 s 무음 → speech → 무음을 실시간 속도로 넣는다. ``rec.inputs`` 에 넣은 음성은 이어서 흘린다."""
    while not stop.is_set() and rec.connected_at is None:
        time.sleep(0.01)
    silence = bytes(FRAME_SIZE_BYTES)
    lead = int(1.0 / FRAME_SEC)
    padded = speech + bytes((-len(speech)) % FRAME_SIZE_BYTES)
    frames = [silence] * lead + [padded[i : i + FRAME_SIZE_BYTES] for i in range(0, len(padded), FRAME_SIZE_BYTES)]
    clip: bytes | None = speech  # frames[clip_at] 을 넣을 때 이 음성이 시작된다 — 녹음용으로 시각을 남긴다
    clip_at = lead
    next_t = time.monotonic()
    i = 0
    while not stop.is_set():
        if i >= len(frames):  # 앞 음성이 끝났다 — 무음 동안 i 는 계속 늘므로 다음 음성은 처음부터 다시 센다
            try:
                more = rec.inputs.get_nowait()
                padded = more + bytes((-len(more)) % FRAME_SIZE_BYTES)
                frames = [padded[j : j + FRAME_SIZE_BYTES] for j in range(0, len(padded), FRAME_SIZE_BYTES)]
                clip, clip_at, i = more, 0, 0
            except queue.Empty:
                pass
        if clip is not None and i == clip_at:
            rec.user_clips.append((time.monotonic(), clip))
            clip = None
        try:
            q.put_nowait(frames[i] if i < len(frames) else silence)
        except queue.Full:
            pass
        i += 1
        next_t += FRAME_SEC
        time.sleep(max(0.0, next_t - time.monotonic()))


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------


def setup(experiment: str, tag: str, subdir: str = "") -> tuple[wiring.ProcessComponents, Path]:
    """기록 훅을 걸고, 로그를 실험 폴더로 보내고, 컴포넌트를 조립한다 (LED 끔, 실험 전용 DB).

    결과 폴더는 ``<번호>_<tag>`` (:mod:`exp_dirs`). ``subdir`` 를 주면 그 아래 — 한 태그에 조건별로 모을 때.
    """
    os.chdir(REPO)
    out_dir = result_dir(experiment, tag) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-28s %(levelname)-5s %(message)s",
        handlers=[logging.FileHandler(out_dir / "pipeline.log")],
    )
    install()
    comps = wiring.build_components(
        db_path=str(out_dir / "experiment.db"),
        device_settings_path=str(out_dir / "device_settings.json"),
        led_enabled=False,
        engine="live",
    )
    return comps, out_dir


def run_session(
    comps: wiring.ProcessComponents,
    rec: Recorder,
    *,
    speech: bytes,
    stop_when: Callable[[Recorder, LiveSessionLoop, float], bool],
    max_sec: float,
    song_sec: float = 2.0,
    wav_path: Path | None = None,
    memory_enabled: bool = False,
) -> LiveSessionLoop:
    """세션 하나를 돈다. ``stop_when(rec, loop, now)`` 이 참이 되거나 연결 후 ``max_sec`` 이 지나면 끊는다.

    노래는 스텁이 ``song_sec`` 뒤에 끝낸다 — 재생 중 요청을 흉내 내려면 길게 준다.
    ``memory_enabled`` 면 운영처럼 장기기억(실험 DB)과 ``search_memory`` 툴을 붙인다.
    """
    global CURRENT
    bridge = StubBridge(song_sec)
    comps.bridge = bridge  # type: ignore[assignment]
    while not comps.audio_queue.empty():
        comps.audio_queue.get_nowait()
    CURRENT = rec
    session = comps.create_session(memory_enabled=memory_enabled)
    session.history.new_session(session.session_id)
    loop = session.session_loop
    assert isinstance(loop, LiveSessionLoop)
    stop = threading.Event()
    threading.Thread(target=_feed, args=(comps.audio_queue, speech, rec, stop), daemon=True).start()

    def watchdog() -> None:
        while not stop.is_set():
            now = time.monotonic()
            if rec.connected_at is not None and (now - rec.connected_at > max_sec or stop_when(rec, loop, now)):
                break
            time.sleep(0.1)
        comps.shutdown_event.set()

    wd = threading.Thread(target=watchdog, daemon=True)
    wd.start()
    try:
        loop.run()
    except Exception:
        logger.exception("session failed")
    stop.set()
    wd.join(timeout=2)
    comps.shutdown_event.clear()
    CURRENT = None
    if wav_path is not None:
        save_conversation_wav(rec, wav_path)
    return loop


def audio_pos(rec: Recorder, t: float) -> float:
    """monotonic 시각 ``t`` 까지 받은 모델 오디오 길이(초) — 녹음·레이블의 시간축."""
    return sum(n for at, n, _ in rec.audio if at <= t) / (BRIDGE_SAMPLE_RATE * 2)


def save_conversation_wav(rec: Recorder, path: Path) -> None:
    """왼쪽 사용자(넣은 음성), 오른쪽 모델(받은 출력)의 스테레오 WAV."""
    clips = [(audio_pos(rec, t), resample(pcm, SAMPLE_RATE, BRIDGE_SAMPLE_RATE)) for t, pcm in rec.user_clips]
    write_conversation_wav(path, bytes(rec.pcm), clips, BRIDGE_SAMPLE_RATE)


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
