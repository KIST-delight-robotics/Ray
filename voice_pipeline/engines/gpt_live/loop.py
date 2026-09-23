"""GPT-Live 엔진의 ACTIVE 세션 루프.

:class:`~voice_pipeline.engines.cascade.loop.SessionLoop` 의 자리를 대신한다 — ASR·턴 감지·LLM·TTS 체인이
모델 하나(gpt-live-1)로 대체되므로, 이 루프가 하는 일은 셋이다.

1. 마이크 프레임을 브리지 레이트로 리샘플해 세션으로 보내고, 모델 출력 오디오를 C++로 그대로 보낸다.
   세션 전체가 ``stream_start`` 하나로 시작하는 스트림이며 무음 조각도 그대로 흘린다(C++ 는
   ``head_motion=False`` 로 대기 모션을 유지하고 입만 움직인다).
2. 전사 세그먼트를 히스토리와 utterances(장기기억 입력)에 저장한다.
3. 백엔드의 함수 툴 호출을 실행한다 — 장기기억 검색 ``search_memory``, 볼륨·밝기 ``adjust_volume`` /
   ``set_brightness`` / ``get_device_settings``, 종료 ``end_conversation``. 핸들러는 executor 에서 돌리고
   프레임 루프는 완료를 폴링한다 — 임베딩·DB 조회가 마이크 전송과 출력 중계를 막지 않게.
4. 노래를 재생한다 (``play_song`` / ``stop_song``, 아래 "노래 재생").
5. 세션 종료를 판정하고 닫는다 — 종료 키워드, 유휴 타임아웃, 백엔드의 ``end_conversation`` 툴,
   세션 만료/연결 끊김, 브리지 오류, 오디오 기아, 외부 stop.

노래 재생 (C++ 는 재생 하나를 끝까지 처리하므로 라이브 스트림과 교대한다):
    ``play_song`` 호출은 결과를 보류한 채(PENDING) 모델 출력이 무음이 되길 기다림 → ``audio_end`` 뒤
    ``playback_complete`` 대기(DRAINING) → 마이크 원본 채널 + 재생 지시 append + ``play_audio_csv``(PLAYING) →
    보류한 툴 결과 제출 → ``playback_complete``(끝) → 마이크·지시 복원, ``stream_start`` 로 재개.
    ``stop_song`` 도 결과를 보류한다(STOPPING): ``stop`` 은 즉시 보내되, 위임 시 모델의 한마디("끌게")가 스트림이
    닫힌 채 끝나길 기다린 뒤 스트림을 열고 결과를 제출한다 — 그래야 "끌게" 는 버려지고 결과 뒤의 "껐어" 만 들린다.
    결과를 노래 시작 뒤에 제출하므로 위임 시 모델의 한마디는 잘리지 않고, 결과 뒤의 말은 스트림이 없어 버려진다.
    노래 상태 동안 백엔드는 ``stop_song`` 하나만 가진 재생용 설정으로 교체된다(``session.update``, 끝나면 복원).
    다른 툴 호출은 오류로 답하고 ``end_conversation`` 도 종료로 치지 않는다. 모델 음성은 버리고 전사는 저장하지 않는다.

종료 시퀀스(실측 근거는 scripts/gpt_live/FINDINGS.md §4~5):
    입력 mute → 모델의 마지막 발화가 끝날 때까지(출력이 무음으로 N초) 대기, 상한 있음 →
    ``session.close``(closed 이벤트는 기다리지 않음) → 브리지 ``audio_end``.
    이후 ``playback_complete`` 대기는 엔트리포인트 FAREWELL 이 맡는다.
"""

from __future__ import annotations

import enum
import json
import logging
import queue
import re
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Executor, Future
from datetime import UTC, datetime
from typing import Any

import numpy as np

from voice_pipeline.adapters.audio_input import AudioInput
from voice_pipeline.adapters.cpp_bridge import CppBridge, CppEventType
from voice_pipeline.adapters.gpt_live import (
    GPTLiveSession,
    LiveAudio,
    LiveClosed,
    LiveError,
    LiveFunctionCall,
    LiveResponseDone,
    LiveTranscript,
    LiveUsage,
)
from voice_pipeline.adapters.led import LEDController, LEDState
from voice_pipeline.engines.gpt_live.instructions import (
    SONG_BACKEND_INSTRUCTIONS,
    SONG_ENDED_INSTRUCTIONS,
    SONG_PLAYING_INSTRUCTIONS,
)
from voice_pipeline.engines.gpt_live.songs import Song
from voice_pipeline.engines.gpt_live.tools import (
    END_CONVERSATION_TOOL,
    PLAY_SONG_TOOL,
    STOP_SONG_TOOL,
    STOP_SONG_TOOL_DEF,
    ToolHandler,
)
from voice_pipeline.history import ConversationHistory
from voice_pipeline.memory.storage import SQLiteMemoryStorage
from voice_pipeline.settings import BRIDGE_SAMPLE_RATE, SAMPLE_RATE
from voice_pipeline.types import AudioFrame, TokenCounter

logger = logging.getLogger("voice_pipeline.live_session")


def _completed(value: str) -> Future[str]:
    """이미 끝난 Future (인라인 실행·오류 경로용)."""
    future: Future[str] = Future()
    future.set_result(value)
    return future


def _rms(pcm: bytes) -> float:
    """16-bit mono PCM 의 RMS."""
    if not pcm:
        return 0.0
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples)))


class Phase(enum.Enum):
    """세션 루프 상태."""

    ACTIVE = "active"  # 대화 진행 중
    ENDING = "ending"  # 종료 결정됨. 입력 mute, 모델의 마지막 발화가 끝나길 기다림
    DONE = "done"


class SongState(enum.Enum):
    """노래 재생 하위 상태 (Phase.ACTIVE 안에서만 NONE 이 아니다)."""

    NONE = "none"
    PENDING = "pending"  # play_song 결과 보류 중. 모델 출력이 무음이 되길 기다림
    DRAINING = "draining"  # audio_end 보냄. 라이브 스트림의 playback_complete 대기
    PLAYING = "playing"  # play_audio_csv 보냄. 노래 끝의 playback_complete 대기
    STOPPING = "stopping"  # stop 보냄. playback_complete 와 모델 출력 무음을 기다린 뒤 스트림 재개


class LiveSessionLoop:
    """GPT-Live 세션 하나를 프레임 루프로 돈다. ``run()`` 이 반환하면 세션이 끝난 것."""

    _FRAME_TIMEOUT_SEC = 0.1  # audio_queue.get 대기
    _MAX_BATCH_FRAMES = 10  # 한 iteration 에 보내는 최대 프레임 수
    _MAX_EVENTS_PER_FRAME = 64  # 한 iteration 에 처리하는 최대 Live 이벤트 수 (오디오 10/s 라 여유 큼)
    _SESSION_TIMEOUT_SEC = 60.0  # 마지막 전사(사용자·모델) 이후 이 시간 지나면 종료
    _AUDIO_STARVATION_TIMEOUT_SEC = 5.0  # 마이크 프레임 단절 → 종료
    _END_MIN_WAIT_SEC = 2.0  # 종료 결정 후 모델이 작별 인사를 시작할 여유
    _END_SILENCE_SEC = 1.0  # 출력 무음이 이만큼 이어지면 마지막 발화가 끝난 것으로 봄
    _END_MAX_WAIT_SEC = 30.0  # 무음이 감지되지 않을 때의 안전망. 정상 발화에서는 걸리지 않아야 한다
    _STREAM_START_MAX_WAIT_SEC = 10.0  # 인사 WAV 의 playback_complete 가 안 와도 이 시간 뒤엔 stream_start
    _SONG_SILENCE_SEC = 1.0  # play_song 뒤 출력 무음이 이만큼 이어지면 스트림을 닫는다
    _SONG_MAX_WAIT_SEC = 30.0  # 무음이 감지되지 않을 때의 안전망. 정상 발화에서는 걸리지 않아야 한다
    _LEAD_LOG_INTERVAL_SEC = 15.0  # 오디오 전송 상태(추정 밀림, 조각 도착 간격) DEBUG 로그 주기
    _GAP_EVENT_SEC = 0.3  # 조각 도착 간격이 이 이상이면 정지 이벤트로 INFO 로그 (C++ [split] 로그와 대조용)
    # 서버 출력은 실시간보다 1~3% 짧게 온다 — 주로 무음 프레임이 빠지고, 늦게라도 오지 않는다
    # (2026-09-16 store 녹음 대조로 확인. 서버 제어 가이드 "Reflected output ranges can have gaps for dropped frames").
    # Wi-Fi 정지가 있으면 더 커진다.
    # C++ 모션 루프는 덩이가 시각보다 먼저 와 있어야 끊기지 않으므로, 보낸 오디오 길이(lead)가 경과 시간보다
    # 이 밴드 이상 뒤지면 무음 조각을 채우고, 네트워크 정지 뒤 몰려와 앞서면 무음 조각을 버려 실시간에 맞춘다.
    # 둘 다 무음 조각(값 전부 0) 뒤에서만 하므로 말소리는 건드리지 않고, 소리와 모션은 같은 스트림을 쓰니 동기 유지.
    _LEAD_PAD_BELOW_SEC = -0.1  # lead 가 이 아래면 무음(정확히 0) 조각 뒤에 채워 _LEAD_PAD_TARGET_SEC 까지 올린다
    _LEAD_PAD_TARGET_SEC = 0.0
    _LEAD_DROP_ABOVE_SEC = 0.3  # lead 가 이 위면 무음(정확히 0) 조각을 버린다
    # 긴 발화 중에는 정확히 0인 조각이 오지 않아 채울 기회가 없다(60초 발화에 여유 500 ms 깎임, 2026-09-15).
    # 어절 사이 쉼(rms 30 미만, 100~300 ms)에서도 채우되, 급할 때만·한 조각만 넣어 쉼이 티 나게 길어지지 않게 한다.
    _QUIET_RMS = 30  # 이 미만이면 말 사이 쉼으로 본다 (16-bit 기준 약 -61 dBFS, 말소리 최약 구간 30~100 보다 아래)
    _LEAD_PAD_IN_SPEECH_BELOW_SEC = -0.3
    # 종료 키워드 — 사용자 세그먼트 텍스트에 부분 문자열로 매칭. "안녕" 은 인사와 겹쳐 제외.
    _EXIT_KEYWORDS: tuple[str, ...] = ("잘 가", "잘가", "이제 갈게", "여기까지", "다음에 봐", "bye", "goodbye")

    def __init__(
        self,
        *,
        live: GPTLiveSession,
        cpp_bridge: CppBridge,
        history: ConversationHistory,
        led: LEDController,
        audio_queue: queue.Queue[AudioFrame],
        memory_storage: SQLiteMemoryStorage | None = None,
        session_id: str | None = None,
        token_counter: TokenCounter | None = None,
        shutdown_event: threading.Event | None = None,
        tool_handlers: dict[str, ToolHandler] | None = None,
        executor: Executor | None = None,
        input_sample_rate: int = SAMPLE_RATE,
        live_sample_rate: int = BRIDGE_SAMPLE_RATE,
        wait_for_playback_complete: bool = False,
        song_catalog: Mapping[str, Song] | None = None,
        audio_input: AudioInput | None = None,
    ) -> None:
        """
        Args:
            live: 시작 전 상태의 GPT-Live 세션. run() 이 start()/close() 를 부른다.
            cpp_bridge: 연결된 C++ 브리지.
            history: 세션 히스토리. 확정된 전사 세그먼트를 user/assistant 메시지로 넣는다.
            led: LED 컨트롤러. 연결되어 마이크가 흐르기 시작하면 IDLE, 입력을 막는 종료 시퀀스부터 SLEEPING.
            wait_for_playback_complete: True 면 브리지가 재생 중인 파일(인사 WAV)의 playback_complete 를 받은 뒤
                stream_start 를 보낸다. 그 전에도 연결·마이크 전송은 하고, 출력 조각은 버린다(모델은 재촉 없으면 침묵).
            audio_queue: AudioInput 이 채우는 마이크 프레임 큐 (input_sample_rate, mono 16-bit).
            memory_storage: 있으면 utterances 를 저장해 MemoryWriter 입력으로 남긴다.
            tool_handlers: 백엔드 함수 툴 이름 → 실행기. ``end_conversation`` 은 내장.
            executor: 툴 핸들러를 돌릴 executor. None 이면 프레임 루프에서 인라인 실행(테스트용).
            input_sample_rate / live_sample_rate: 마이크 레이트와 세션 레이트. 다르면 선형 보간으로 리샘플.
            song_catalog: 재생 가능한 노래 (키 → Song). None/빈 dict 면 play_song 은 unknown 으로 답한다.
            audio_input: 있으면 노래 재생 중 원본 마이크 채널로 바꾼다 (:meth:`AudioInput.set_raw_capture`).
        """
        self._live = live
        self._bridge = cpp_bridge
        self._history = history
        self._led = led
        self._audio_queue = audio_queue
        self._memory_storage = memory_storage
        self._session_id = session_id
        self._token_counter = token_counter
        self._shutdown_event = shutdown_event
        self._tool_handlers: dict[str, ToolHandler] = dict(tool_handlers or {})
        self._tool_handlers.setdefault(END_CONVERSATION_TOOL, lambda _args: json.dumps({"status": "ending"}))
        self._executor = executor
        self._in_rate = input_sample_rate
        self._live_rate = live_sample_rate
        self._wait_for_playback_complete = wait_for_playback_complete
        self._song_catalog: Mapping[str, Song] = dict(song_catalog or {})
        self._audio_input = audio_input
        self._song_state = SongState.NONE
        self._song: Song | None = None
        self._song_future: Future[str] | None = None  # 보류 중인 play_song 결과. 노래 시작(또는 취소) 시 완료
        self._stop_future: Future[str] | None = None  # 보류 중인 stop_song 결과. 스트림 재개 시 완료
        self._song_requested_time = 0.0
        self._stop_requested_time = 0.0
        self._song_playback_done = False  # STOPPING 중 playback_complete 를 받았는가
        self._song_cancelled = False  # DRAINING 중 stop 이 오면 재생 대신 스트림만 다시 연다
        self._stream_start_pending = False
        self._stream_start_deadline = 0.0
        self._discarded_before_stream_sec = 0.0

        self._stop_event = threading.Event()
        self._phase = Phase.ACTIVE
        self._exit_reason = ""
        self._stream_open = False
        self._pending_calls: list[LiveFunctionCall] = []  # 응답 완료 전까지 모은 함수 호출
        self._inflight: list[tuple[LiveFunctionCall, Future[str]]] = []  # 실행 중인 핸들러
        self._last_frame_time = 0.0
        self._last_transcript_time = 0.0
        self._last_voice_time: float | None = None
        self._end_started_time = 0.0
        self._audio_sent_sec = 0.0
        self._first_audio_time: float | None = None
        self._first_voice_logged = False
        self._last_audio_time: float | None = None
        self._max_audio_gap_sec = 0.0  # 로그 구간 내 조각 도착 최대 간격 (정상 0.1 s)
        self._max_audio_gap_total_sec = 0.0  # 세션 전체 최대 간격 (종료 요약용)
        self._padded_sec = 0.0  # 채워 넣은 무음 누적
        self._dropped_sec = 0.0  # 버린 무음 누적
        self._last_lead_log_time = 0.0
        self._context_ratio: float | None = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def exit_reason(self) -> str:
        """세션이 끝난 이유 (로그·테스트용)."""
        return self._exit_reason

    def run(self) -> None:
        """세션을 시작하고 끝날 때까지 돈다. 예외는 세션을 닫은 뒤 다시 던진다."""
        self._start_session()
        try:
            while not self._run_frame():
                pass
        except Exception:
            self._exit_reason = self._exit_reason or "error"
            logger.error("LiveSessionLoop failed", exc_info=True)
            self._finish(graceful=False)
            raise
        else:
            self._finish(graceful=True)

    def request_stop(self) -> None:
        """외부에서 세션 종료를 요청한다 (다음 프레임에 종료 시퀀스 진입)."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _start_session(self) -> None:
        self._phase = Phase.ACTIVE
        self._exit_reason = ""
        logger.info("Connecting to GPT-Live…")
        t0 = time.monotonic()
        started = self._live.start()
        remain_min = (started.expires_at - time.time()) / 60.0
        elapsed = time.monotonic() - t0
        logger.info("GPT-Live connected in %.1fs: %s (expires in %.0f min)", elapsed, started.session_id, remain_min)
        self._led.set_state(LEDState.IDLE)  # 마이크가 세션으로 흐르기 시작 = 대화 가능

        now = time.monotonic()
        self._last_frame_time = now
        self._last_transcript_time = now
        self._last_lead_log_time = now
        if self._wait_for_playback_complete:
            self._stream_start_pending = True
            self._stream_start_deadline = now + self._STREAM_START_MAX_WAIT_SEC
            logger.info("LiveSessionLoop started (listening; stream_start after greeting playback)")
        else:
            self._open_stream()
            logger.info("LiveSessionLoop started (stream_start sent)")

    def _open_stream(self) -> None:
        """C++ 출력 스트림을 연다. 세션 전체가 스트림 하나 — live 표시로 C++ 가 프리버퍼를 모은 뒤 시작한다."""
        self._bridge.send_stream_start(live=True)
        self._stream_open = True
        self._stream_start_pending = False
        # lead 회계는 스트림에 실제로 보낸 첫 조각부터
        self._first_audio_time = None
        self._last_audio_time = None
        if self._discarded_before_stream_sec > 0:
            logger.info("stream_start sent — discarded %.1fs of output while closed", self._discarded_before_stream_sec)

    def _finish(self, *, graceful: bool) -> None:
        if self._phase == Phase.DONE:
            return
        self._phase = Phase.DONE
        for _, future in self._inflight:
            future.cancel()  # 아직 시작 안 한 것만 취소된다. 실행 중인 결과는 버려진다
        self._inflight.clear()
        if self._song_state is SongState.PLAYING:
            try:
                self._bridge.send_stop()
            except Exception:
                logger.warning("stop for playing song failed", exc_info=True)
        self._set_raw_capture(False)
        self._clear_song("cancelled")
        try:
            self._live.close(graceful=graceful)
        except Exception:
            logger.warning("Live close failed", exc_info=True)
        if self._stream_open:
            self._stream_open = False
            try:
                self._bridge.send_audio_end()
            except Exception:
                logger.warning("audio_end send failed", exc_info=True)
        logger.info(
            "Audio summary: sent %.0fs, padded %.1fs, dropped %.1fs, max chunk gap %.0fms",
            self._audio_sent_sec,
            self._padded_sec,
            self._dropped_sec,
            self._max_audio_gap_total_sec * 1000,
        )
        logger.info("LiveSessionLoop ended (%s)", self._exit_reason or "unknown")

    # ------------------------------------------------------------------
    # Frame loop
    # ------------------------------------------------------------------

    def _run_frame(self) -> bool:
        """한 iteration. True 를 돌려주면 루프 종료."""
        if self._shutdown_event is not None and self._shutdown_event.is_set():
            self._exit_reason = self._exit_reason or "shutdown"
            return True
        if self._stop_event.is_set() and self._phase == Phase.ACTIVE:
            self._begin_ending("stop_requested")

        # 1. 마이크 → Live
        if not self._pump_microphone():
            return True

        # 2. Live 이벤트
        for _ in range(self._MAX_EVENTS_PER_FRAME):
            event = self._live.poll_event()
            if event is None:
                break
            if self._handle_live_event(event):
                return True
        self._poll_tools()

        # 3. 브리지 이벤트 — 오류는 예외로 올라와 run() 이 세션을 닫는다
        while (cpp_event := self._bridge.poll_event()) is not None:
            if cpp_event.event_type != CppEventType.PLAYBACK_COMPLETE:
                continue
            if self._stream_start_pending:
                self._open_stream()  # 인사 WAV 끝 → 이제 출력 스트림을 연다
                continue
            if self._song_state is SongState.DRAINING:
                self._start_song()  # 라이브 스트림이 다 나갔다 → 노래 시작
                continue
            if self._song_state is SongState.PLAYING:
                self._end_song()  # 노래 끝 → 대화 재개
                continue
            if self._song_state is SongState.STOPPING:
                self._song_playback_done = True  # 모델의 한마디가 끝나면 타이머가 _end_song
                continue
            # 우리가 audio_end 를 보내기 전에 왔다 = C++ 가 스트림을 끝냈다(stop 등). 다시 열지 않고 종료.
            logger.warning("playback_complete before audio_end — bridge stream ended")
            self._stream_open = False
            self._exit_reason = self._exit_reason or "bridge_stream_ended"
            return True

        # 4. 타이머
        now = time.monotonic()
        if self._stream_start_pending and now > self._stream_start_deadline:
            logger.warning("Greeting playback_complete not received — opening stream anyway")
            self._open_stream()
        if self._song_state is SongState.PENDING and self._utterance_finished(
            self._song_requested_time,
            now,
            min_wait=0.0,
            silence=self._SONG_SILENCE_SEC,
            max_wait=self._SONG_MAX_WAIT_SEC,
            label="Song",
        ):
            self._drain_for_song()
        if (
            self._song_state is SongState.STOPPING
            and self._song_playback_done
            and self._utterance_finished(
                self._stop_requested_time,
                now,
                min_wait=0.0,
                silence=self._SONG_SILENCE_SEC,
                max_wait=self._SONG_MAX_WAIT_SEC,
                label="Song stop",
            )
        ):
            self._end_song()
        idle = now - self._last_transcript_time > self._SESSION_TIMEOUT_SEC
        if self._phase == Phase.ACTIVE and self._song_state is SongState.NONE and idle:
            self._begin_ending("idle_timeout")
        if self._phase == Phase.ENDING and self._ending_complete(now):
            return True
        self._maybe_log_lead(now)
        return False

    def _pump_microphone(self) -> bool:
        """마이크 프레임을 세션으로 보낸다. False 면 오디오 기아로 종료."""
        try:
            frame = self._audio_queue.get(timeout=self._FRAME_TIMEOUT_SEC)
        except queue.Empty:
            starving = time.monotonic() - self._last_frame_time > self._AUDIO_STARVATION_TIMEOUT_SEC
            if self._phase == Phase.ACTIVE and starving:
                logger.error("Audio starvation — ending session")
                self._exit_reason = "audio_starvation"
                return False
            return True

        frames = [frame]
        for _ in range(self._MAX_BATCH_FRAMES - 1):
            try:
                frames.append(self._audio_queue.get_nowait())
            except queue.Empty:
                break
        self._last_frame_time = time.monotonic()
        # 종료 시퀀스 중에도 계속 보낸다 — 입력이 끊기면 서버 타임라인이 멈춘다(FINDINGS §1 서드파티 보고)
        self._live.send_audio(self._resample(b"".join(frames)))
        return True

    # ------------------------------------------------------------------
    # Live events
    # ------------------------------------------------------------------

    def _handle_live_event(self, event: Any) -> bool:
        """이벤트 하나를 처리한다. True 면 루프 종료."""
        if isinstance(event, LiveAudio):
            self._on_audio(event.pcm)
        elif isinstance(event, LiveTranscript):
            self._on_transcript(event)
        elif isinstance(event, LiveFunctionCall):
            self._pending_calls.append(event)
        elif isinstance(event, LiveResponseDone):
            self._on_response_done()
        elif isinstance(event, LiveUsage):
            self._context_ratio = event.context_ratio
            logger.debug("Live usage %.0fs, context %.1f%%", event.seconds, (event.context_ratio or 0.0) * 100)
        elif isinstance(event, LiveClosed):
            logger.info("Live session closed by server: %s", event.reason)
            self._exit_reason = self._exit_reason or f"live_closed:{event.reason}"
            return True
        elif isinstance(event, LiveError):
            logger.warning("Live error %s: %s", event.code, event.message)
        return False

    def _on_audio(self, pcm: bytes) -> None:
        now = time.monotonic()
        if not self._stream_open:
            # 인사 WAV 재생 중이거나 노래 중: 출력이 갈 스트림이 없다. 음성 유무만 추적한다(무음 대기용)
            self._discarded_before_stream_sec += len(pcm) / (self._live_rate * 2)
            if any(pcm) and _rms(pcm) >= self._QUIET_RMS:
                self._last_voice_time = now
            return
        if self._first_audio_time is None:
            self._first_audio_time = now
        if self._last_audio_time is not None:
            gap = now - self._last_audio_time
            self._max_audio_gap_sec = max(self._max_audio_gap_sec, gap)
            self._max_audio_gap_total_sec = max(self._max_audio_gap_total_sec, gap)
            if gap >= self._GAP_EVENT_SEC:
                speaking = self._last_voice_time is not None and self._last_audio_time - self._last_voice_time < 0.5
                state = "speaking" if speaking else "silent"
                logger.info("Audio chunk gap %.0fms (%s, %s)", gap * 1000, state, self._phase.name)
        self._last_audio_time = now

        chunk_sec = len(pcm) / (self._live_rate * 2)
        silent = not any(pcm)  # 서버는 말하지 않을 때 값이 전부 0인 조각을 보낸다
        quiet = silent or _rms(pcm) < self._QUIET_RMS  # 어절 사이 쉼, 세션 시작 직후의 미세 잡음(진폭 <50) 포함
        if not quiet:
            self._last_voice_time = now
            if not self._first_voice_logged:
                self._first_voice_logged = True
                logger.info("Model started speaking")

        lead = self._audio_sent_sec - (now - self._first_audio_time)
        if silent and lead > self._LEAD_DROP_ABOVE_SEC:
            self._dropped_sec += chunk_sec  # 앞서 있으면 무음 조각은 버린다
            return
        self._bridge.send_audio(pcm)
        self._audio_sent_sec += chunk_sec
        lead += chunk_sec
        if silent and lead < self._LEAD_PAD_BELOW_SEC:
            self._pad_silence(len(pcm), count=int((self._LEAD_PAD_TARGET_SEC - lead) / chunk_sec + 0.999))
        elif quiet and lead < self._LEAD_PAD_IN_SPEECH_BELOW_SEC:
            self._pad_silence(len(pcm), count=1)  # 말 사이 쉼: 한 조각만

    def _pad_silence(self, chunk_bytes: int, *, count: int) -> None:
        """무음 조각 count 개를 채워 보낸다 (소리·모션이 같은 스트림을 쓰므로 동기는 유지)."""
        zeros = bytes(chunk_bytes)
        chunk_sec = chunk_bytes / (self._live_rate * 2)
        for _ in range(max(count, 0)):
            self._bridge.send_audio(zeros)
            self._audio_sent_sec += chunk_sec
            self._padded_sec += chunk_sec

    def _on_transcript(self, seg: LiveTranscript) -> None:
        self._last_transcript_time = time.monotonic()
        can_exit = self._phase == Phase.ACTIVE and self._song_state is SongState.NONE  # 노래 중엔 정지만 받는다
        if seg.speaker == "user" and can_exit and self._matches_exit_keyword(seg.text):
            logger.info("Exit keyword in user speech: %r", seg.text)
            self._begin_ending("exit_keyword")
        if not seg.closed or not seg.text.strip():
            return
        text = seg.text.strip()
        if self._song_state in (SongState.PLAYING, SongState.STOPPING):
            logger.info("%s (during song, not stored): %s", seg.speaker, text)
            return
        logger.info("%s: %s", seg.speaker, text)
        if seg.speaker == "user":
            self._history.add_user_message(text)
        else:
            self._history.add_assistant_message(text)
        self._save_utterance(seg.speaker, text)

    def _on_response_done(self) -> None:
        """백엔드 응답이 끝났다 — 모아둔 함수 호출을 실행에 넘긴다. 결과 제출은 :meth:`_poll_tools`."""
        if not self._pending_calls:
            return
        calls, self._pending_calls = self._pending_calls, []
        for call in calls:
            logger.info("Tool %s(%s) requested", call.name, call.arguments)
            self._inflight.append((call, self._submit_tool(call)))
        self._poll_tools()  # 인라인 실행이면 여기서 바로 끝난다. executor 면 프레임 루프가 이어서 폴링

    def _submit_tool(self, call: LiveFunctionCall) -> Future[str]:
        if call.name == STOP_SONG_TOOL:
            return self._request_stop()  # 루프 내장. 스트림이 다시 열린 뒤 완료되는 Future
        if self._song_state is not SongState.NONE:
            return _completed(json.dumps({"error": "a song is playing; only stop_song is available"}))
        if call.name == PLAY_SONG_TOOL:
            return self._request_song(call)  # 루프 내장. 노래가 시작된 뒤 완료되는 Future
        handler = self._tool_handlers.get(call.name)
        if handler is None:
            logger.warning("Backend requested unknown tool %s", call.name)
            return _completed(json.dumps({"error": f"unknown function {call.name}"}))
        if self._executor is None:
            try:
                return _completed(handler(call.arguments))
            except Exception as exc:
                future: Future[str] = Future()
                future.set_exception(exc)
                return future
        return self._executor.submit(handler, call.arguments)

    def _poll_tools(self) -> None:
        """실행 중인 핸들러가 전부 끝났으면 결과를 제출하고 백엔드 응답을 이어간다(또는 종료 시퀀스)."""
        if not self._inflight or not all(future.done() for _, future in self._inflight):
            return
        calls, self._inflight = self._inflight, []
        if self._phase != Phase.ACTIVE:
            # 종료 시퀀스 중 도착한 결과는 버린다 — 이어가면 모델이 새 발화를 시작해 종료가 늦어진다
            logger.info("Dropping %d tool result(s) — session is %s", len(calls), self._phase.name)
            return
        ending = False
        stopped = False
        for call, future in calls:
            try:
                output = future.result()
            except Exception as exc:
                output = json.dumps({"error": str(exc)})
                logger.warning("Tool %s failed", call.name, exc_info=True)
            logger.info("Tool %s -> %s", call.name, output if len(output) <= 200 else output[:200] + "…")
            self._live.submit_function_output(call.call_id, output)
            ending |= call.name == END_CONVERSATION_TOOL and self._song_state is SongState.NONE
            stopped |= call.name == STOP_SONG_TOOL and '"stopped"' in output
        if ending:
            # 이어가면 백엔드가 작별 인사를 또 만들고 모델이 두 번 말할 수 있다(FINDINGS §4). 위임은 미완으로 둔다.
            self._begin_ending("end_conversation_tool")
        else:
            self._live.continue_response()
            if stopped:
                self._restore_backend()  # 정지 결과의 후속 응답은 재생용 백엔드로 만들어진 뒤에 복원

    # ------------------------------------------------------------------
    # Ending
    # ------------------------------------------------------------------

    def _begin_ending(self, reason: str) -> None:
        if self._phase != Phase.ACTIVE:
            return
        self._phase = Phase.ENDING
        self._exit_reason = reason
        self._end_started_time = time.monotonic()
        logger.info("Ending session (%s) — muting input, waiting for last utterance", reason)
        self._led.set_state(LEDState.SLEEPING)  # 입력을 막는 순간부터 대화 불가
        if self._song_state in (SongState.PLAYING, SongState.DRAINING, SongState.PENDING):
            logger.info("Song %s cancelled by ending", self._song_state.value)
            self._stop_song()  # 끊긴 노래의 playback_complete 뒤 _end_song 이 스트림을 다시 연다
        try:
            self._live.mute_input()
        except Exception:
            logger.warning("mute_input failed", exc_info=True)

    def _ending_complete(self, now: float) -> bool:
        return self._utterance_finished(
            self._end_started_time,
            now,
            min_wait=self._END_MIN_WAIT_SEC,
            silence=self._END_SILENCE_SEC,
            max_wait=self._END_MAX_WAIT_SEC,
            label="Ending",
        )

    def _utterance_finished(
        self, started: float, now: float, *, min_wait: float, silence: float, max_wait: float, label: str
    ) -> bool:
        """``started`` 이후 모델 발화가 끝났는가.

        최소 대기(``min_wait``) 뒤 출력 무음이 ``silence`` 만큼 이어지면 끝으로 보고, ``max_wait`` 가 상한.
        """
        since_start = now - started
        if since_start >= max_wait:
            logger.warning("%s: no output silence for %.0fs — forcing on; check silence detection", label, max_wait)
            return True
        if since_start < min_wait:
            return False
        quiet_since = max(self._last_voice_time or 0.0, started)
        if now - quiet_since >= silence:
            logger.info("%s: output silent for %.1fs", label, now - quiet_since)
            return True
        return False

    # ------------------------------------------------------------------
    # Song playback
    # ------------------------------------------------------------------

    def _request_song(self, call: LiveFunctionCall) -> Future[str]:
        """play_song 요청. 곡이 유효하면 결과를 보류하고 PENDING 으로 들어간다."""
        try:
            key = str(json.loads(call.arguments or "{}").get("song", ""))
        except json.JSONDecodeError:
            key = ""
        song = self._song_catalog.get(key)
        if song is None:
            return _completed(json.dumps({"error": f"unknown song {key!r}"}))
        if self._song_state is not SongState.NONE:
            return _completed(json.dumps({"error": "a song is already playing"}))
        self._song = song
        self._song_future = Future()
        self._song_state = SongState.PENDING
        self._song_requested_time = time.monotonic()
        try:
            self._live.update_backend(SONG_BACKEND_INSTRUCTIONS, (STOP_SONG_TOOL_DEF,))
        except Exception:
            logger.warning("backend update for song failed", exc_info=True)
        logger.info("Song requested: %s — waiting for output silence", song.key)
        return self._song_future

    def _request_stop(self) -> Future[str]:
        """stop_song 요청. 노래 상태면 결과를 보류하고 정지 절차를 시작한다."""
        if self._song_state is SongState.NONE:
            return _completed(json.dumps({"status": "not_playing"}))
        if self._stop_future is not None:
            return _completed(json.dumps({"status": "stopping"}))
        self._stop_future = Future()
        self._stop_song()
        return self._stop_future

    def _resolve_song_future(self, status: str) -> None:
        if self._song_future is None or self._song_future.done():
            return
        song = self._song
        payload = {"status": status}
        if song is not None:
            payload.update(song=song.key, title=song.title, artist=song.artist)
        self._song_future.set_result(json.dumps(payload))

    def _clear_song(self, status: str) -> None:
        """노래 상태를 지우고 보류한 결과를 완료한다. 정지 툴이 없었으면 백엔드도 여기서 복원한다."""
        had_song = self._song_state is not SongState.NONE
        self._resolve_song_future(status)
        if self._stop_future is not None and not self._stop_future.done():
            self._stop_future.set_result(json.dumps({"status": "stopped"}))
        stop_pending_submit = self._stop_future is not None and self._phase is Phase.ACTIVE
        self._song_future = None
        self._stop_future = None
        self._song_state = SongState.NONE
        self._song = None
        self._song_playback_done = False
        if had_song and not stop_pending_submit and self._phase is not Phase.DONE:
            self._restore_backend()  # 정지 결과가 제출될 예정이면 _poll_tools 가 그 뒤에 복원한다

    def _restore_backend(self) -> None:
        try:
            self._live.restore_backend()
        except Exception:
            logger.warning("backend restore after song failed", exc_info=True)

    def _drain_for_song(self) -> None:
        """라이브 스트림을 닫고 playback_complete 를 기다린다."""
        if self._stream_open:
            self._stream_open = False
            self._bridge.send_audio_end()
        self._stream_start_pending = False  # 인사 중이면 인사의 playback_complete 가 DRAINING 을 끝낸다
        self._song_state = SongState.DRAINING
        logger.info("Song: live stream closed, waiting for drain")

    def _start_song(self) -> None:
        assert self._song is not None
        if self._song_cancelled:
            self._song_cancelled = False
            logger.info("Song %s cancelled before start — reopening stream", self._song.key)
            self._end_song()
            return
        self._set_raw_capture(True)
        try:
            self._live.append_instructions(SONG_PLAYING_INSTRUCTIONS)
        except Exception:
            logger.warning("append for song failed", exc_info=True)
        self._bridge.send_play_audio_csv(self._song.key)
        self._song_state = SongState.PLAYING
        self._resolve_song_future("playing")  # 이제 결과를 제출한다 — 백엔드 완료 뒤 모델 말은 스트림이 없어 버려진다
        logger.info("Song playing: %s", self._song.key)

    def _stop_song(self) -> None:
        if self._song_state is SongState.PLAYING:
            logger.info("Song stop requested")
            self._bridge.send_stop()
            self._song_state = SongState.STOPPING
            self._stop_requested_time = time.monotonic()
            self._song_playback_done = False
        elif self._song_state is SongState.DRAINING:
            self._song_cancelled = True
        elif self._song_state is SongState.PENDING:
            logger.info("Song cancelled before start")
            self._clear_song("cancelled")

    def _end_song(self) -> None:
        key = self._song.key if self._song else "?"
        self._clear_song("cancelled")  # 정상 시작된 노래면 play 결과는 이미 완료돼 있어 무시된다
        self._set_raw_capture(False)
        try:
            self._live.append_instructions(SONG_ENDED_INSTRUCTIONS)
        except Exception:
            logger.warning("append after song failed", exc_info=True)
        self._last_transcript_time = time.monotonic()  # 유휴 타임아웃 재시작
        self._open_stream()
        logger.info("Song ended: %s — stream reopened", key)

    def _set_raw_capture(self, enabled: bool) -> None:
        if self._audio_input is None:
            return
        try:
            self._audio_input.set_raw_capture(enabled)
        except Exception:
            logger.warning("set_raw_capture(%s) failed", enabled, exc_info=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _matches_exit_keyword(self, text: str) -> bool:
        normalized = re.sub(r"[^\w\s가-힣]", " ", text.lower())
        return any(kw in normalized for kw in self._EXIT_KEYWORDS)

    def _resample(self, pcm: bytes) -> bytes:
        if self._in_rate == self._live_rate:
            return pcm
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        n_out = len(samples) * self._live_rate // self._in_rate
        x_in = np.linspace(0.0, 1.0, len(samples), endpoint=False)
        x_out = np.linspace(0.0, 1.0, n_out, endpoint=False)
        return np.interp(x_out, x_in, samples).astype(np.int16).tobytes()

    def _save_utterance(self, role: str, text: str) -> None:
        if self._memory_storage is None or self._session_id is None:
            return
        try:
            timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
            token_count = self._token_counter(text) if self._token_counter else 0
            self._memory_storage.add_utterance(self._session_id, role, text, timestamp, token_count)
        except Exception:
            logger.warning("Failed to save %s utterance", role, exc_info=True)

    def _maybe_log_lead(self, now: float) -> None:
        """보낸 오디오 길이 − 첫 오디오 이후 경과 시간. 양수가 커지면 C++ 쪽에 밀림이 쌓인 것."""
        if self._first_audio_time is None or now - self._last_lead_log_time < self._LEAD_LOG_INTERVAL_SEC:
            return
        self._last_lead_log_time = now
        lead = self._audio_sent_sec - (now - self._first_audio_time)
        gap_ms = self._max_audio_gap_sec * 1000
        self._max_audio_gap_sec = 0.0
        ctx = f"{self._context_ratio * 100:.1f}%" if self._context_ratio is not None else "?"
        # lead: 보낸 오디오 − 경과 시간 (정상 0 근처, 음수면 서버가 뒤처짐). gap: 조각 도착 최대 간격 (정상 100 ms).
        msg = "Audio lead %.2fs, max chunk gap %.0fms (sent %.0fs, padded %.1fs, dropped %.1fs), context %s"
        logger.debug(msg, lead, gap_ms, self._audio_sent_sec, self._padded_sec, self._dropped_sec, ctx)
