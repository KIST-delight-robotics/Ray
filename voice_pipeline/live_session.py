"""GPT-Live 엔진의 ACTIVE 세션 루프.

:class:`~voice_pipeline.session_loop.SessionLoop` 의 자리를 대신한다 — ASR·턴 감지·LLM·TTS 체인이
모델 하나(gpt-live-1)로 대체되므로, 이 루프가 하는 일은 셋이다.

1. 마이크 프레임을 브리지 레이트로 리샘플해 세션으로 보내고, 모델 출력 오디오를 C++로 그대로 보낸다.
   세션 전체가 ``stream_start`` 하나로 시작하는 스트림이며 무음 조각도 그대로 흘린다(C++ 는
   ``head_motion=False`` 로 대기 모션을 유지하고 입만 움직인다).
2. 전사 세그먼트를 히스토리와 utterances(장기기억 입력)에 저장한다.
3. 세션 종료를 판정하고 닫는다 — 종료 키워드, 유휴 타임아웃, 백엔드의 ``end_conversation`` 툴,
   세션 만료/연결 끊김, 브리지 오류, 오디오 기아, 외부 stop.

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
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np

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
from voice_pipeline.history import ConversationHistory
from voice_pipeline.memory.storage import SQLiteMemoryStorage
from voice_pipeline.settings import BRIDGE_SAMPLE_RATE, SAMPLE_RATE
from voice_pipeline.types import AudioFrame, TokenCounter

logger = logging.getLogger("voice_pipeline.live_session")


def _rms(pcm: bytes) -> float:
    """16-bit mono PCM 의 RMS."""
    if not pcm:
        return 0.0
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples)))


# ---------------------------------------------------------------------------
# Prompts — 공식 프롬프팅 가이드 템플릿 구조(역할 / 백채널 / 인터럽트 / 위임 정책 3라벨).
# 규칙을 덧붙이는 식으로 쓰면 위임 판단이 흔들린다(FINDINGS §4: 템플릿 6/6 vs 덧붙임 3/5).
# ---------------------------------------------------------------------------

DEFAULT_LIVE_INSTRUCTIONS = """\
You are Ray, a small, friendly desk robot.
Speak Korean unless the user asks to switch. Keep a casual, warm tone and short answers.
If the user is frustrated, acknowledge it briefly and focus on the next helpful step.

Backchannel policy: Use moderate backchannels. Acknowledge naturally without competing with the main response.

Interruption policy: Stop speaking when the user interrupts. Listen to what they say.

Delegation policy:
Backend tools:
- Web search: current date and time, weather, news, and facts you are not sure about.
- End of conversation: closes the session when the user is done talking.
Delegate to the backend when:
- The request needs current information or a fact you are not sure about.
- The user says goodbye or wants to end the conversation. Say a short goodbye yourself at the same time.
Do not delegate to the backend when:
- You can answer from the conversation or a still-current result.
- The user is chatting, greeting, or thinking aloud.
Delegate before giving an answer that depends on backend work.
Do not guess the result while waiting. Do not mention the backend.
"""

DEFAULT_BACKEND_INSTRUCTIONS = """\
You are the backend for Ray, a Korean-speaking desk robot.

Tools:
- Use web search for current information.
- Call end_conversation when the user says goodbye or wants to stop, then reply with an empty message.

Answer in Korean, in one or two short sentences that sound natural when spoken aloud.
No lists, no URLs, no markdown.
"""

LIVE_VOICE = "marin"  # 세션 목소리. 인사 WAV 도 같은 목소리로 합성해 이질감을 없앤다
LIVE_GREETING_TEXT = "네, 부르셨어요?"  # 웨이크워드 뒤 세션 연결 지연(1.5~3.5초)을 가리는 인사
LIVE_GREETING_TTS_MODEL = "gpt-4o-mini-tts"  # Live 목소리(marin)를 지원하는 OpenAI TTS 모델

END_CONVERSATION_TOOL = "end_conversation"

DEFAULT_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "name": END_CONVERSATION_TOOL,
        "description": (
            "End the current conversation session. "
            "Call this when the user says goodbye or clearly wants to stop talking."
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
)

ToolHandler = Callable[[str], str]  # arguments(JSON 문자열) → output(JSON 문자열)


class Phase(enum.Enum):
    """세션 루프 상태."""

    ACTIVE = "active"  # 대화 진행 중
    ENDING = "ending"  # 종료 결정됨. 입력 mute, 모델의 마지막 발화가 끝나길 기다림
    DONE = "done"


class LiveSessionLoop:
    """GPT-Live 세션 하나를 프레임 루프로 돈다. ``run()`` 이 반환하면 세션이 끝난 것."""

    _FRAME_TIMEOUT_SEC = 0.1  # audio_queue.get 대기
    _MAX_BATCH_FRAMES = 10  # 한 iteration 에 보내는 최대 프레임 수
    _MAX_EVENTS_PER_FRAME = 64  # 한 iteration 에 처리하는 최대 Live 이벤트 수 (오디오 10/s 라 여유 큼)
    _SESSION_TIMEOUT_SEC = 60.0  # 마지막 전사(사용자·모델) 이후 이 시간 지나면 종료
    _AUDIO_STARVATION_TIMEOUT_SEC = 5.0  # 마이크 프레임 단절 → 종료
    _END_MIN_WAIT_SEC = 2.0  # 종료 결정 후 모델이 작별 인사를 시작할 여유
    _END_SILENCE_SEC = 1.0  # 출력 무음이 이만큼 이어지면 마지막 발화가 끝난 것으로 봄
    _END_MAX_WAIT_SEC = 8.0  # 무음이 안 와도 종료하는 상한
    _STREAM_START_MAX_WAIT_SEC = 10.0  # 인사 WAV 의 playback_complete 가 안 와도 이 시간 뒤엔 stream_start
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
        input_sample_rate: int = SAMPLE_RATE,
        live_sample_rate: int = BRIDGE_SAMPLE_RATE,
        wait_for_playback_complete: bool = False,
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
            input_sample_rate / live_sample_rate: 마이크 레이트와 세션 레이트. 다르면 선형 보간으로 리샘플.
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
        self._in_rate = input_sample_rate
        self._live_rate = live_sample_rate
        self._wait_for_playback_complete = wait_for_playback_complete
        self._stream_start_pending = False
        self._stream_start_deadline = 0.0
        self._discarded_before_stream_sec = 0.0

        self._stop_event = threading.Event()
        self._phase = Phase.ACTIVE
        self._exit_reason = ""
        self._stream_open = False
        self._pending_calls: list[LiveFunctionCall] = []
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
            logger.info("stream_start sent — discarded %.1fs of output before it", self._discarded_before_stream_sec)

    def _finish(self, *, graceful: bool) -> None:
        if self._phase == Phase.DONE:
            return
        self._phase = Phase.DONE
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

        # 3. 브리지 이벤트 — 오류는 예외로 올라와 run() 이 세션을 닫는다
        while (cpp_event := self._bridge.poll_event()) is not None:
            if cpp_event.event_type != CppEventType.PLAYBACK_COMPLETE:
                continue
            if self._stream_start_pending:
                self._open_stream()  # 인사 WAV 끝 → 이제 출력 스트림을 연다
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
        if self._phase == Phase.ACTIVE and now - self._last_transcript_time > self._SESSION_TIMEOUT_SEC:
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
        if self._stream_start_pending:
            # 인사 WAV 재생 중: 스트림이 아직 없다
            self._discarded_before_stream_sec += len(pcm) / (self._live_rate * 2)
            return
        now = time.monotonic()
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
        if seg.speaker == "user" and self._phase == Phase.ACTIVE and self._matches_exit_keyword(seg.text):
            logger.info("Exit keyword in user speech: %r", seg.text)
            self._begin_ending("exit_keyword")
        if not seg.closed or not seg.text.strip():
            return
        text = seg.text.strip()
        logger.info("%s: %s", seg.speaker, text)
        if seg.speaker == "user":
            self._history.add_user_message(text)
        else:
            self._history.add_assistant_message(text)
        self._save_utterance(seg.speaker, text)

    def _on_response_done(self) -> None:
        if not self._pending_calls:
            return
        calls, self._pending_calls = self._pending_calls, []
        ending = False
        for call in calls:
            handler = self._tool_handlers.get(call.name)
            if handler is None:
                output = json.dumps({"error": f"unknown function {call.name}"})
                logger.warning("Backend requested unknown tool %s", call.name)
            else:
                try:
                    output = handler(call.arguments)
                except Exception as exc:
                    output = json.dumps({"error": str(exc)})
                    logger.warning("Tool %s failed", call.name, exc_info=True)
            logger.info("Tool %s(%s) -> %s", call.name, call.arguments, output)
            self._live.submit_function_output(call.call_id, output)
            ending |= call.name == END_CONVERSATION_TOOL
        if ending:
            # 이어가면 백엔드가 작별 인사를 또 만들고 모델이 두 번 말할 수 있다(FINDINGS §4). 위임은 미완으로 둔다.
            self._begin_ending("end_conversation_tool")
        else:
            self._live.continue_response()

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
        try:
            self._live.mute_input()
        except Exception:
            logger.warning("mute_input failed", exc_info=True)

    def _ending_complete(self, now: float) -> bool:
        since_start = now - self._end_started_time
        if since_start >= self._END_MAX_WAIT_SEC:
            logger.info("Ending: max wait reached")
            return True
        if since_start < self._END_MIN_WAIT_SEC:
            return False
        quiet_since = max(self._last_voice_time or 0.0, self._end_started_time)
        if now - quiet_since >= self._END_SILENCE_SEC:
            logger.info("Ending: output silent for %.1fs", now - quiet_since)
            return True
        return False

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
