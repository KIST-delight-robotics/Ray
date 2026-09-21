"""GPT-Live 경로 오디오·모션 싱크 프로브 — 가짜 live 세션으로 합성 톤 버스트를 실시간 페이싱해 보내고,
스피커 출력(마이크 녹음)과 입 모터 실제 위치(C++ Standard_Log)의 온셋을 대조해 고정 오프셋과 드리프트를 잰다.

    uv run python scripts/bench/live_sync_probe.py --duration 180            # 실행 + 분석
    uv run python scripts/bench/live_sync_probe.py --analyze var/log/sync_probe/<ts>   # 분석만

전제: build/Ray 가 떠 있어야 한다 (RAY_UNIT 필요). 실제 LiveSessionLoop·CppBridge 를 그대로 쓰고 GPTLiveSession 자리만
FakeLiveSession 으로 바꾼다 — Python 의 lead 채움·버림과 C++ 프리버퍼·split 채움 경로가 프로덕션과 동일하게 돈다.

신호: ``--period`` 마다 ``--burst-sec`` 길이의 배음 톤 버스트(Hann 창), 나머지는 정확히 0 (서버의 무음 조각과 같음).
주입: ``--stall-every/--stall-ms`` (Wi-Fi 정지 모사: 조각을 붙잡다 한 번에 방출), ``--shortfall`` (서버 프레임 유실 모사: 무음 조각 누락).

시간축: C++ 는 cycle 0 에서 ``playback_started`` 를 보내며 ``start_time`` 을 잡는다. 브리지 수신 스레드가 그 이벤트를 큐에 넣는
순간의 monotonic 을 기준점으로 삼고, 녹음의 첫 read 시각과 Standard_Log(RESPONSES 행 = audio_sync 행과 같은 틱에 기록) 을
같은 축으로 옮긴다. 고정 오프셋에는 마이크 입력 지연(USB, 수십 ms)이 포함된다 — 드리프트·계단 판정에는 무관.

출력 (var/log/sync_probe/<ts>/): capture_6ch.wav, meta.json, probe.log, offsets.csv, report.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import random
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from voice_pipeline.adapters.cpp_bridge import CppBridge, CppEventType  # noqa: E402
from voice_pipeline.adapters.gpt_live import LiveAudio, LiveStarted  # noqa: E402
from voice_pipeline.engines.gpt_live.loop import LiveSessionLoop  # noqa: E402
from voice_pipeline.settings import BRIDGE_SAMPLE_RATE, FRAME_SIZE_SAMPLES, SAMPLE_RATE, SAMPLE_WIDTH  # noqa: E402

LOG_ROOT = Path("var/log/sync_probe")
MOTION_LOG_ROOT = Path("var/log/motion")
CHUNK_SEC = 0.1  # GPT-Live 관측 조각 길이
CAPTURE_CH = 6
CAPTURE_DEVICE_HINTS = ("respeaker", "respeaker_hw", "default")

logger = logging.getLogger("sync_probe")


# ---------------------------------------------------------------------------
# Fake live session
# ---------------------------------------------------------------------------


class FakeLiveSession:
    """GPTLiveSession 과 같은 표면. 출력 조각을 실시간 속도로 만들어 poll_event 로 내준다."""

    def __init__(
        self,
        *,
        period: float,
        burst_sec: float,
        amplitude: float,
        stall_every: float,
        stall_ms: float,
        shortfall: float,
        seed: int = 0,
    ) -> None:
        self._period = period
        self._burst_sec = burst_sec
        self._amplitude = amplitude
        self._stall_every = stall_every
        self._stall_ms = stall_ms
        self._shortfall = shortfall
        self._rng = random.Random(seed)
        self._events: queue.Queue[LiveAudio] = queue.Queue()
        self._stop = threading.Event()
        self._muted = threading.Event()
        self._thread: threading.Thread | None = None
        self._n = int(BRIDGE_SAMPLE_RATE * CHUNK_SEC)
        self._zeros = bytes(self._n * 2)
        self.first_emit_at: float | None = None
        self.chunks_emitted = 0
        self.chunks_dropped = 0
        self.stalls = 0

    @property
    def session_id(self) -> str:
        return "fake-live"

    def start(self) -> LiveStarted:
        self._thread = threading.Thread(target=self._produce, name="fake-live", daemon=True)
        self._thread.start()
        return LiveStarted(session_id=self.session_id, expires_at=time.time() + 7200)

    def close(self, *, graceful: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def send_audio(self, pcm: bytes) -> None:
        pass

    def mute_input(self) -> None:
        self._muted.set()  # 이후 무음만 — 루프의 종료 판정(출력 무음 1초)이 성립하도록

    def submit_function_output(self, call_id: str, output: str) -> None:
        pass

    def continue_response(self) -> None:
        pass

    def poll_event(self) -> LiveAudio | None:
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    def _chunk(self, index: int) -> bytes:
        # 버스트는 스트림 시각 period/2 에서 시작해 period 마다 반복
        t = (index * self._n + np.arange(self._n)) / BRIDGE_SAMPLE_RATE
        phase = np.mod(t - self._period / 2, self._period)
        mask = phase < self._burst_sec
        if not mask.any():
            return self._zeros
        env = 0.5 * (1.0 - np.cos(2 * np.pi * phase / self._burst_sec)) * mask
        tone = (
            np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 660 * t)
        ) / 1.75
        return (env * tone * self._amplitude * 32767).astype(np.int16).tobytes()

    def _produce(self) -> None:
        t0 = time.monotonic()
        self.first_emit_at = t0
        index = 0
        held: list[bytes] = []
        stall_end: float | None = None
        next_stall_at = t0 + self._stall_every if self._stall_every > 0 else None
        while not self._stop.is_set():
            due = t0 + index * CHUNK_SEC
            now = time.monotonic()
            if due > now:
                time.sleep(min(due - now, 0.05))
                continue
            pcm = self._zeros if self._muted.is_set() else self._chunk(index)
            index += 1
            silent = pcm is self._zeros
            if self._shortfall > 0 and silent and self._rng.random() < self._shortfall:
                self.chunks_dropped += 1
                continue
            if stall_end is not None:
                held.append(pcm)
                if now >= stall_end:
                    for h in held:
                        self._events.put(LiveAudio(h))
                    self.chunks_emitted += len(held)
                    held.clear()
                    stall_end = None
                    self.stalls += 1
                    next_stall_at = now + self._stall_every
                continue
            if next_stall_at is not None and now >= next_stall_at and not self._muted.is_set():
                stall_end = now + self._stall_ms / 1000.0
                held.append(pcm)
                continue
            self._events.put(LiveAudio(pcm))
            self.chunks_emitted += 1


# ---------------------------------------------------------------------------
# Helpers around the real components
# ---------------------------------------------------------------------------


class _StampingQueue(queue.Queue):
    """브리지 수신 스레드가 넣는 순간의 시각을 이벤트별로 기록한다 (프레임 루프 폴링 지연 배제)."""

    def __init__(self) -> None:
        super().__init__()
        self.stamps: list[tuple[str, float]] = []

    def put(self, item, block=True, timeout=None):  # noqa: ANN001
        et = getattr(item, "event_type", None)
        if isinstance(et, CppEventType):
            self.stamps.append((et.value, time.monotonic()))
        super().put(item, block, timeout)


class MicFeeder(threading.Thread):
    """LiveSessionLoop 의 마이크 펌프가 기아로 끝나지 않도록 30 ms 0 프레임을 넣는다."""

    def __init__(self, q: queue.Queue) -> None:
        super().__init__(name="mic-feeder", daemon=True)
        self._q = q
        self.stop_event = threading.Event()

    def run(self) -> None:
        frame = bytes(FRAME_SIZE_SAMPLES * SAMPLE_WIDTH)
        period = FRAME_SIZE_SAMPLES / SAMPLE_RATE
        nxt = time.monotonic()
        while not self.stop_event.is_set():
            self._q.put(frame)
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))


class Capture(threading.Thread):
    """reSpeaker 6ch 녹음. first_read_at = 첫 프레임의 첫 샘플이 들어온 monotonic (추정)."""

    def __init__(self, device_index: int | None) -> None:
        super().__init__(name="capture", daemon=True)
        self._device_index = device_index
        self.chunks: list[bytes] = []
        self.read_at: list[float] = []  # 각 read 완료 monotonic — 장치 클럭 대 monotonic 비율 추정용
        self.first_read_at: float | None = None
        self.device_name = ""
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.error: Exception | None = None

    def run(self) -> None:
        import pyaudio

        pa = pyaudio.PyAudio()
        try:
            st, self.device_name = self._open(pa)
            while not self.stop_event.is_set():
                data = st.read(FRAME_SIZE_SAMPLES, exception_on_overflow=False)
                now = time.monotonic()
                if self.first_read_at is None:
                    self.first_read_at = now - FRAME_SIZE_SAMPLES / SAMPLE_RATE
                    self.ready.set()
                self.read_at.append(now)
                self.chunks.append(data)
            st.stop_stream()
            st.close()
        except Exception as exc:  # noqa: BLE001
            self.error = exc
            self.ready.set()
        finally:
            pa.terminate()

    def _open(self, pa):  # noqa: ANN001, ANN202
        candidates: list[int] = []
        if self._device_index is not None:
            candidates.append(self._device_index)
        for hint in CAPTURE_DEVICE_HINTS:
            for i in range(pa.get_device_count()):
                d = pa.get_device_info_by_index(i)
                if hint in d["name"].lower() and d["maxInputChannels"] >= CAPTURE_CH and i not in candidates:
                    candidates.append(i)
        last: Exception | None = None
        for i in candidates:
            try:
                st = pa.open(
                    format=pa.get_format_from_width(SAMPLE_WIDTH),
                    channels=CAPTURE_CH,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=i,
                    frames_per_buffer=FRAME_SIZE_SAMPLES,
                )
                return st, pa.get_device_info_by_index(i)["name"]
            except Exception as exc:  # noqa: BLE001
                last = exc
                logger.warning("capture device %d open failed: %s", i, exc)
        raise RuntimeError(f"no capture device opened (tried {candidates}): {last}")


def _latest_motion_dir(after: float) -> Path | None:
    best: tuple[float, Path] | None = None
    for d in MOTION_LOG_ROOT.iterdir() if MOTION_LOG_ROOT.exists() else []:
        f = d / "audio_sync_RESPONSES.csv"
        if f.exists() and f.stat().st_mtime >= after and (best is None or f.stat().st_mtime > best[0]):
            best = (f.stat().st_mtime, d)
    return best[1] if best else None


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> Path:
    run_dir = LOG_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    fh = logging.FileHandler(run_dir / "probe.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    logging.getLogger("voice_pipeline.live_session").setLevel(logging.DEBUG)

    LiveSessionLoop._SESSION_TIMEOUT_SEC = args.duration + 300  # 전사가 없으니 유휴 종료를 막는다
    wall_start = time.time()

    bridge = CppBridge()
    bridge.connect()
    stamps = _StampingQueue()
    bridge._event_queue = stamps  # 수신 스레드는 호출 시점에 속성을 읽는다
    logger.info("bridge connected")

    cap = Capture(args.device)
    cap.start()
    cap.ready.wait(timeout=5.0)
    if cap.error is not None:
        raise SystemExit(f"capture failed: {cap.error}")
    logger.info("capture started on %r", cap.device_name)

    fake = FakeLiveSession(
        period=args.period,
        burst_sec=args.burst_sec,
        amplitude=args.amplitude,
        stall_every=args.stall_every,
        stall_ms=args.stall_ms,
        shortfall=args.shortfall,
    )
    audio_queue: queue.Queue = queue.Queue()
    feeder = MicFeeder(audio_queue)
    feeder.start()

    loop = LiveSessionLoop(
        live=fake,
        cpp_bridge=bridge,
        history=MagicMock(),
        led=MagicMock(),
        audio_queue=audio_queue,
        token_counter=len,
    )
    timer = threading.Timer(args.duration, loop.request_stop)
    timer.daemon = True
    timer.start()
    logger.info(
        "running for %.0fs (period %.2fs, burst %.0fms, amp %.2f, stall %s/%sms, shortfall %.3f)",
        args.duration,
        args.period,
        args.burst_sec * 1000,
        args.amplitude,
        args.stall_every,
        args.stall_ms,
        args.shortfall,
    )
    try:
        loop.run()
    finally:
        timer.cancel()
        feeder.stop_event.set()
        time.sleep(1.5)  # C++ 꼬리 재생·playback_complete 여유
        cap.stop_event.set()
        cap.join(timeout=3.0)
        bridge.disconnect()

    with wave.open(str(run_dir / "capture_6ch.wav"), "wb") as w:
        w.setnchannels(CAPTURE_CH)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(b"".join(cap.chunks))

    motion_dir = _latest_motion_dir(wall_start - 60)
    meta = {
        "wall_start": wall_start,
        "duration_sec": args.duration,
        "period": args.period,
        "burst_sec": args.burst_sec,
        "amplitude": args.amplitude,
        "stall_every": args.stall_every,
        "stall_ms": args.stall_ms,
        "shortfall": args.shortfall,
        "capture_device": cap.device_name,
        "capture_first_read_at": cap.first_read_at,
        "capture_read_at": cap.read_at,
        "capture_rate": SAMPLE_RATE,
        "capture_channels": CAPTURE_CH,
        "analysis_channel": args.channel,
        "bridge_stamps": stamps.stamps,
        "fake_first_emit_at": fake.first_emit_at,
        "fake_chunks_emitted": fake.chunks_emitted,
        "fake_chunks_dropped": fake.chunks_dropped,
        "fake_stalls": fake.stalls,
        "py_audio_sent_sec": loop._audio_sent_sec,
        "py_padded_sec": loop._padded_sec,
        "py_dropped_sec": loop._dropped_sec,
        "py_max_gap_sec": loop._max_audio_gap_total_sec,
        "motion_dir": str(motion_dir) if motion_dir else None,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    logger.info("run saved to %s (motion logs: %s)", run_dir, motion_dir)
    return run_dir


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

ANALYSIS_FS = 100.0  # 공통 시간축 (Hz)
TONE_HZ = (220.0, 440.0, 660.0)  # FakeLiveSession 의 배음 — 마이크에서 이 성분만 추적해 모터 소음을 배제


def _tone_envelope(x: np.ndarray, fs: int, out_fs: float) -> tuple[np.ndarray, np.ndarray]:
    """톤 주파수 성분의 크기(Goertzel 식) 를 out_fs 로 뽑는다. 반환: (시각(초), 크기)."""
    win = int(0.04 * fs)
    hop = int(round(fs / out_fs))
    n = (x.size - win) // hop
    idx = np.arange(win)
    kernels = np.stack([np.exp(-2j * np.pi * f * idx / fs) * np.hanning(win) for f in TONE_HZ])
    frames = np.lib.stride_tricks.as_strided(x, shape=(n, win), strides=(x.strides[0] * hop, x.strides[0]))
    mag = np.abs(frames @ kernels.T)  # (n, 3)
    env = mag @ np.array([1.0, 0.5, 0.25])
    t = (np.arange(n) * hop + win / 2) / fs
    return t, env


def _template(t: np.ndarray, period: float, burst_sec: float) -> np.ndarray:
    """스트림 시각 t 에서의 이상적 버스트 엔벨로프 (FakeLiveSession._chunk 와 같은 Hann 창)."""
    phase = np.mod(t - period / 2, period)
    return np.where(phase < burst_sec, 0.5 * (1 - np.cos(2 * np.pi * phase / burst_sec)), 0.0)


def _signed_open(pos: np.ndarray) -> np.ndarray:
    """모터 위치 → 입 열림량(닫힘=0, 양수). 닫힌 위치가 대부분이라 중앙값을 기준으로 잡는다."""
    d = pos - np.median(pos)
    if abs(d.min()) > abs(d.max()):
        d = -d
    return np.clip(d, 0, None)


def _xcorr_lag(a: np.ndarray, b: np.ndarray, fs: float, max_lag: float) -> tuple[float, float]:
    """b 가 a 보다 얼마나 늦는지(초). 정규화 상호상관 최대 위치 + 포물선 보간. 반환: (lag, 상관계수)."""
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan"), 0.0
    m = int(max_lag * fs)
    lags = np.arange(-m, m + 1)
    c = np.array([np.dot(a[max(0, -k) : a.size - max(0, k)], b[max(0, k) : b.size - max(0, -k)]) for k in lags]) / (
        na * nb
    )
    k = int(np.argmax(c))
    if 0 < k < c.size - 1:
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]
        denom = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        frac = 0.0
    return float((lags[k] + frac) / fs), float(c[k])


def _windowed_lags(
    t: np.ndarray, a: np.ndarray, b: np.ndarray, fs: float, win_sec: float, max_lag: float
) -> list[tuple[float, float, float]]:
    """win_sec 창마다 (창 중심 시각, lag, corr)."""
    out = []
    n = int(win_sec * fs)
    for s0 in range(0, a.size - n + 1, n):
        lag, corr = _xcorr_lag(a[s0 : s0 + n], b[s0 : s0 + n], fs, max_lag)
        out.append((float(t[s0] + win_sec / 2), lag, corr))
    return out


def _fit(t: np.ndarray, y: np.ndarray) -> float:
    if t.size < 2:
        return float("nan")
    return float(np.polyfit(t, y, 1)[0])


def _fmt_lags(rows: list[tuple[float, float, float]], label: str) -> list[str]:
    lines = [f"  {label}:"]
    for tc, lag, corr in rows:
        lines.append(f"    t={tc:6.0f}s  {lag * 1000:+7.1f} ms  (r={corr:.2f})")
    return lines


def analyze(run_dir: Path) -> str:
    meta = json.loads((run_dir / "meta.json").read_text())
    period, burst_sec = meta["period"], meta["burst_sec"]
    lines: list[str] = [f"# live sync probe — {run_dir}", ""]

    started = [t for name, t in meta["bridge_stamps"] if name == "playback_started"]
    if not started:
        raise SystemExit("playback_started 이벤트가 기록되지 않았다 — C++ 스트림이 시작되지 않은 실행")
    t_start = started[0]  # C++ start_time ≈ 스트림 위치 0 이 재생기로 들어간 시각

    # --- 마이크: 톤 성분 엔벨로프 (C++ start_time 축) ---
    with wave.open(str(run_dir / "capture_6ch.wav"), "rb") as w:
        fs = w.getframerate()
        nch = w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, nch)
    x = raw[:, meta["analysis_channel"]].astype(np.float64) / 32768.0
    t_env, env = _tone_envelope(x, fs, ANALYSIS_FS)
    t_env = t_env + (meta["capture_first_read_at"] - t_start)
    snr = np.percentile(env, 99) / max(np.percentile(env, 50), 1e-12)
    lines.append(
        f"capture: {meta['capture_device']} ch{meta['analysis_channel']}, {raw.shape[0] / fs:.0f}s, tone p99/p50 = {snr:.0f}"
    )
    # 장치 클럭 대 monotonic: read 완료 시각 = a + b·(누적 샘플/fs). b−1 > 0 이면 장치 클럭이 느리다.
    # XVF3800 은 ADC·DAC 가 한 클럭이라 이 값이 곧 재생 드리프트이고, 마이크 축도 같이 늘어나 물리 측정에서는 상쇄된다.
    dev_ppm = float("nan")
    reads = np.asarray(meta.get("capture_read_at") or [])
    if reads.size > 100:
        nominal = np.arange(1, reads.size + 1) * FRAME_SIZE_SAMPLES / fs
        b = np.polyfit(nominal, reads, 1)[0]
        dev_ppm = (b - 1.0) * 1e6
        lines.append(
            f"device clock vs monotonic: {dev_ppm:+.1f} ppm (= {dev_ppm * 60 / 1000:+.2f} ms/min; + = 장치 느림, 재생이 뒤처진다)"
        )

    # --- 모터: audio_sync 행 ↔ Standard_Log RESPONSES 행 (같은 틱) ---
    mdir = Path(meta["motion_dir"]) if meta.get("motion_dir") else None
    if mdir is None or not (mdir / "audio_sync_RESPONSES.csv").exists():
        raise SystemExit("motion 로그 디렉터리를 찾지 못했다")
    sync = np.genfromtxt(mdir / "audio_sync_RESPONSES.csv", delimiter=",", names=True)
    # Standard_Log 는 Ray 프로세스당 하나에 세션이 이어 붙고(사이에 WAIT 행), 살아 있는 동안 꼬리가 플러시되지 않을 수
    # 있다. 마지막 RESPONSES 연속 구간의 완전한 행(21열)만 쓴다 — audio_sync CSV 는 세션마다 새로 쓰이므로 그와 짝이다.
    segments: list[list[list[str]]] = []
    for ln in (mdir / "Standard_Log.csv").read_text().splitlines()[1:]:
        r = ln.split(",")
        if len(r) == 21 and r[1] == "RESPONSES":
            if not segments or segments[-1] is None:
                segments.append([])
            segments[-1].append(r)
        elif segments and segments[-1] is not None:
            segments.append(None)  # 구간 경계 표시
    std_rows = next((seg for seg in reversed(segments) if seg), [])
    n = min(len(std_rows), sync.size)
    std_ts = np.array([float(r[0]) for r in std_rows[:n]])
    base = std_ts - sync["elapsed_ms"][:n]
    if len(std_rows) != sync.size or np.ptp(base) > 5.0:
        lines.append(
            f"warn: Standard_Log rows {len(std_rows)} vs audio_sync rows {sync.size}; clock base spread {np.ptp(base):.1f} ms"
        )
    t_tick = sync["elapsed_ms"][:n] / 1000.0
    target_open = _signed_open(np.array([float(r[10]) for r in std_rows[:n]]))
    present_open = _signed_open(np.array([float(r[15]) for r in std_rows[:n]]))
    lines.append(
        f"motor: {mdir.name}, ticks {n}, mouth open p99 {np.percentile(present_open, 99):.0f} dxl (target {np.percentile(target_open, 99):.0f})"
    )

    # --- 공통 축 (start_time 기준, 겹치는 구간) ---
    t0 = max(t_env[0], t_tick[0]) + 2.0  # 시작 블렌딩 구간 제외
    t1 = min(t_env[-1], t_tick[-1])
    tu = np.arange(t0, t1, 1.0 / ANALYSIS_FS)
    env_u = np.interp(tu, t_env, env)
    pres_u = np.interp(tu, t_tick, present_open)
    targ_u = np.interp(tu, t_tick, target_open)
    tmpl_u = _template(tu, period, burst_sec)
    # 엔벨로프를 입 대역(40 ms 틱)에 맞춰 부드럽게
    k = int(0.04 * ANALYSIS_FS)
    env_s = np.convolve(env_u, np.ones(k) / k, mode="same")
    tmpl_s = np.convolve(tmpl_u, np.ones(k) / k, mode="same")

    win_sec = 30.0
    max_lag = period / 2 * 0.95
    mouth_vs_audio = _windowed_lags(tu, env_s, pres_u, ANALYSIS_FS, win_sec, max_lag)  # +: 입이 소리보다 늦다
    present_vs_target = _windowed_lags(tu, targ_u, pres_u, ANALYSIS_FS, win_sec, max_lag)
    audio_vs_stream = _windowed_lags(tu, tmpl_s, env_s, ANALYSIS_FS, win_sec, max_lag)  # 스트림 위치 → 마이크 도달
    target_vs_stream = _windowed_lags(tu, tmpl_s, targ_u, ANALYSIS_FS, win_sec, max_lag)  # 스트림 위치 → 입 목표

    def summary(rows: list[tuple[float, float, float]]) -> tuple[float, float, float]:
        tc = np.array([r[0] for r in rows]) / 60.0
        lag = np.array([r[1] for r in rows]) * 1000
        return (
            float(lag[0]) if lag.size else float("nan"),
            _fit(tc, lag),
            float(np.median([r[2] for r in rows])) if rows else 0.0,
        )

    lines.append("")
    first, slope, rmed = summary(mouth_vs_audio)
    lines.append(
        f"MOUTH vs AUDIO (physical, primary): first window {first:+.1f} ms, drift {slope:+.2f} ms/min over {(t1 - t0) / 60:.1f} min, median r {rmed:.2f}"
    )
    lines.append("  (+ = 입 실제 위치가 스피커 소리보다 늦음. 절대값에는 마이크 입력 지연이 들어감)")
    lines += _fmt_lags(mouth_vs_audio, "per window")
    if not np.isnan(dev_ppm):
        lines.append(
            f"  → 물리 드리프트 추정 (마이크 축 클럭 보정): {slope - dev_ppm * 60 / 1000:+.2f} ms/min "
            f"(측정 {slope:+.2f} − 장치 클럭 {dev_ppm * 60 / 1000:+.2f}; 음수 = 시간이 갈수록 소리가 입보다 늦어짐)"
        )
    first, slope, rmed = summary(present_vs_target)
    lines.append(f"motor mechanical lag (present vs target): {first:+.1f} ms, drift {slope:+.2f} ms/min, r {rmed:.2f}")
    stream_ok = meta["py_padded_sec"] == 0 and meta["py_dropped_sec"] == 0
    lines.append("")
    lines.append(
        "chain latencies vs stream position"
        + ("" if stream_ok else " (Python pad/drop 가 있어 스트림 축이 밀렸음 — 참고만)")
    )
    first, slope, rmed = summary(audio_vs_stream)
    lines.append(
        f"  stream → mic: {first:+.1f} ms, drift {slope:+.2f} ms/min, r {rmed:.2f}   (출력 지연 + 마이크 입력 지연)"
    )
    first, slope, rmed = summary(target_vs_stream)
    lines.append(
        f"  stream → mouth target: {first:+.1f} ms, drift {slope:+.2f} ms/min, r {rmed:.2f}   (음수면 입이 선행)"
    )

    # --- 내부 대조: audio_sync CSV ---
    diff = sync["expected_ms"] - (sync["playing_ms"] - sync["silence_ms"])
    late = sync["elapsed_ms"] - sync["expected_ms"]
    good = np.abs(diff) < 30
    s_int = _fit(sync["elapsed_ms"][good] / 60000.0, diff[good])
    lines.append("")
    lines.append(
        f"internal (audio_sync): expected−content {diff.min():+.0f}..{diff.max():+.0f} ms, slope {s_int:+.2f} ms/min, "
        f"underrun silence {sync['silence_ms'][-1]:.0f} ms, tick lateness max {late.max():.0f} ms, ticks>300ms late {(late > 300).sum()}"
    )
    console = mdir / "console.log"
    if console.exists():
        txt = console.read_text(errors="replace").splitlines()
        pads = [ln for ln in txt if "[split]" in ln and "padded" in ln and "stream done" not in ln]
        trims = [ln for ln in txt if "[split]" in ln and "trimmed" in ln and "stream done" not in ln]
        unders = [ln for ln in txt if "underrun: inserted" in ln]
        done = [ln for ln in txt if "stream done" in ln]
        lines.append(
            f"cpp console: split pad events {len(pads)}, trim events {len(trims)}, underrun inserts {len(unders)}"
            + (f"\n  {done[-1].strip()}" if done else "")
        )
    lines.append(
        f"python: sent {meta['py_audio_sent_sec']:.0f}s, padded {meta['py_padded_sec']:.1f}s, dropped {meta['py_dropped_sec']:.1f}s, "
        f"max chunk gap {meta['py_max_gap_sec'] * 1000:.0f} ms | fake: emitted {meta['fake_chunks_emitted']}, dropped {meta['fake_chunks_dropped']}, stalls {meta['fake_stalls']}"
    )

    np.savetxt(
        run_dir / "lags.csv",
        np.array(
            [
                [r[0], r[1] * 1000, r[2], q[1] * 1000, a[1] * 1000, m[1] * 1000]
                for r, q, a, m in zip(mouth_vs_audio, present_vs_target, audio_vs_stream, target_vs_stream, strict=True)
            ]
        ),
        delimiter=",",
        header="t_sec,mouth_minus_audio_ms,corr,present_minus_target_ms,stream_to_mic_ms,stream_to_target_ms",
        comments="",
        fmt="%.3f",
    )
    report = "\n".join(lines)
    (run_dir / "report.txt").write_text(report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration", type=float, default=180.0, help="송신 시간(초). 이후 종료 시퀀스(무음 1초) 진입")
    p.add_argument("--period", type=float, default=1.44, help="버스트 간격(초)")
    p.add_argument("--burst-sec", type=float, default=0.3, help="버스트 길이(초)")
    p.add_argument("--amplitude", type=float, default=0.4, help="버스트 진폭(0~1 full scale)")
    p.add_argument("--stall-every", type=float, default=0.0, help="정지 주입 간격(초), 0 = 없음")
    p.add_argument("--stall-ms", type=float, default=800.0, help="정지 길이(ms)")
    p.add_argument("--shortfall", type=float, default=0.0, help="무음 조각 누락 확률 (서버 유실 모사)")
    p.add_argument("--device", type=int, default=None, help="캡처 장치 인덱스 (기본: respeaker 6ch 자동)")
    p.add_argument("--channel", type=int, default=2, help="분석에 쓸 캡처 채널 (2~5 = 원본 마이크)")
    p.add_argument("--analyze", type=Path, default=None, help="이 실행 디렉터리만 분석")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_dir = args.analyze or run(args)
    print()
    print(analyze(run_dir))


if __name__ == "__main__":
    main()
