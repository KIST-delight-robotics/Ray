"""OpenAI GPT-Live (Live API) 세션 래퍼.

full-duplex 음성 모델과의 WebSocket 세션 하나를 감싼다. 오디오를 보내고, 모델의 오디오·전사·
위임 이벤트를 프로젝트 타입(:class:`LiveEvent`)으로 바꿔 큐에 넣는다. 세션 루프가
:meth:`poll_event` 로 꺼내 쓴다 — :class:`~voice_pipeline.adapters.cpp_bridge.CppBridge` 와 같은 패턴.

Threading model:
    호출자 스레드가 start()/send_*()/append_*()/close() 를 부르고, 데몬 수신 스레드가 SDK
    ``recv()`` 를 돌려 이벤트를 큐에 넣는다. 전사 조각은 SDK ``TranscriptGrouper`` 로 화자별
    세그먼트로 묶는데, 이 그루퍼는 비활성 타이머를 별도 스레드에서 돌리므로 콜백은 큐에 넣는 것만 한다.

API 사실:
    - 오디오 포맷·목소리·instructions·위임 모드는 세션 시작 후 바꿀 수 없다. 추가는 append 3종만.
    - 서버는 말하지 않을 때도 값이 0인 조각을 실시간 속도로 계속 보낸다.
    - 위임 이벤트에는 요청 텍스트가 없다. responses 위임이면 백엔드가 대화 이력 전체를 받는다.
    - 앱이 ``response.create`` 로 백엔드를 부르면 서버가 그것도 위임으로 감싼다 — 대화 모델이 받는 말을 하고,
      결과 반환과 이어가기까지 해야 위임이 닫힌다.
    - instructions.append 는 진행 중 발화를 끊는다. thinking.append 는 끊지 않는다.
"""

from __future__ import annotations

import base64
import json
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from voice_pipeline import trace
from voice_pipeline.settings import BRIDGE_SAMPLE_RATE

logger = logging.getLogger("voice_pipeline.gpt_live")


# ---------------------------------------------------------------------------
# Events (프로젝트 쪽에 노출되는 타입 — SDK 타입은 이 파일 밖으로 나가지 않는다)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveStarted:
    """세션이 열렸다. ``expires_at`` 은 epoch 초 (실측 120분 뒤)."""

    session_id: str
    expires_at: float


@dataclass(frozen=True)
class LiveAudio:
    """모델 출력 오디오 한 조각. ``BRIDGE_SAMPLE_RATE`` mono 16-bit PCM. 무음이면 전부 0."""

    pcm: bytes


@dataclass(frozen=True)
class LiveTranscript:
    """화자별로 묶인 전사 세그먼트의 스냅샷.

    ``closed`` 가 False 면 진행 중(텍스트가 누적되어 다시 온다), True 면 확정. ``text`` 는 항상
    세그먼트 전체 텍스트(델타가 아님). 시각은 세션 타임라인 ms.
    """

    segment_id: str
    speaker: Literal["user", "assistant"]
    text: str
    start_ms: int
    end_ms: int
    closed: bool
    reason: str = ""


@dataclass(frozen=True)
class LiveFunctionCall:
    """responses 위임 백엔드가 요청한 함수 호출. 결과는 :meth:`GPTLiveSession.submit_function_output`."""

    call_id: str
    name: str
    arguments: str
    delegation_id: str | None
    response_id: str | None = None


@dataclass(frozen=True)
class LiveDelegationCreated:
    """위임이 생겼다. 앱이 보낸 요청이 만든 위임이면 ``client_event_id`` 가 그 요청의 event_id."""

    delegation_id: str | None
    client_event_id: str | None = None


@dataclass(frozen=True)
class LiveResponseDone:
    """백엔드 응답 하나가 끝났다. 모아둔 함수 호출 결과를 제출하고 이어갈 시점."""

    delegation_id: str | None
    response_id: str | None = None
    status: str = "unknown"  # completed / failed / incomplete


@dataclass(frozen=True)
class LiveUsage:
    """누적 사용량. ``context_ratio`` 는 컨텍스트 창 사용 비율(0~1), 모르면 None."""

    seconds: float
    context_ratio: float | None


@dataclass(frozen=True)
class LiveClosed:
    """세션이 끝났다. ``reason`` 은 서버 사유 또는 ``connection_lost``."""

    reason: str


@dataclass(frozen=True)
class LiveError:
    """서버가 보낸 오류 이벤트. 세션은 계속될 수 있다."""

    code: str
    message: str


LiveEvent = (
    LiveStarted
    | LiveAudio
    | LiveTranscript
    | LiveDelegationCreated
    | LiveFunctionCall
    | LiveResponseDone
    | LiveUsage
    | LiveClosed
    | LiveError
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveSessionConfig:
    """세션 시작 시 한 번 보내는 설정. 시작 후에는 바꿀 수 없다.

    Attributes:
        instructions: 대화 모델 지시문 (페르소나, 언어, 위임 정책). 16,384 토큰 이하.
        backend_model: responses 위임 백엔드 모델. None 이면 client 위임(백엔드 없음).
        backend_instructions: 백엔드 모델 지시문 (툴 사용 규칙, 답 형식).
        tools: 백엔드에 주는 function 툴 정의(Responses API function 형식). web_search 는 항상 포함.
        backend_reasoning: 백엔드 추론 노력(reasoning.effort). 운영은 wiring 이 low 를 넘긴다.
        voice: 목소리 이름. 기본 marin.
        sample_rate: 입출력 공유 PCM 레이트. 16000 또는 24000.
    """

    instructions: str
    backend_model: str | None = None
    backend_instructions: str = ""
    tools: tuple[dict[str, Any], ...] = ()
    backend_reasoning: str = "none"
    voice: str = "marin"
    sample_rate: int = BRIDGE_SAMPLE_RATE
    model: str = "gpt-live-1"

    def to_session_dict(self) -> dict[str, Any]:
        """SDK ``session.start`` 에 넣을 dict."""
        cfg: dict[str, Any] = {
            "model": self.model,
            "instructions": self.instructions,
            "audio": {
                "format": {"type": "audio/pcm", "rate": self.sample_rate},
                "output": {"voice": self.voice},
            },
        }
        if self.backend_model is not None:
            cfg["delegation"] = {
                "type": "responses",
                "responses": {
                    "model": self.backend_model,
                    "instructions": self.backend_instructions,
                    "tools": [{"type": "web_search"}, *self.tools],
                    "reasoning": {"effort": self.backend_reasoning},
                },
            }
        return cfg


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

ConnectionFactory = Callable[[], Any]


class GPTLiveSession:
    """GPT-Live 세션 하나. start() → send_audio()/poll_event() 반복 → close().

    Args:
        config: 세션 설정.
        connection_factory: 테스트용. SDK ``LiveConnection`` 과 같은 표면(``recv``, ``session.*``,
            ``response.*``, ``close``)을 가진 객체를 돌려주는 호출자. None 이면 ``openai.OpenAI().live.connect()``.
    """

    _START_TIMEOUT_SEC = 15.0  # session.started 대기 상한. 실측 1.5~3.5초
    _START_ATTEMPTS = 3  # 연결·session.started 실패 시 재시도 횟수
    _START_RETRY_DELAY_SEC = 1.0
    _RECV_JOIN_TIMEOUT_SEC = 3.0  # close 시 수신 스레드 종료 대기

    def __init__(self, config: LiveSessionConfig, *, connection_factory: ConnectionFactory | None = None) -> None:
        self._config = config
        self._connection_factory = connection_factory or self._default_connection_factory
        self._conn: Any = None
        self._events: queue.Queue[LiveEvent] = queue.Queue()
        self._receiver: threading.Thread | None = None
        self._stop = threading.Event()
        self._closed_seen = threading.Event()
        self._lock = threading.Lock()
        self._error: RuntimeError | None = None
        self._grouper: Any = None
        self._session_id: str | None = None
        self._active_responses: dict[str | None, str] = {}  # delegation_id → 진행 중 response_id
        self._delegation_created_at: dict[str | None, float] = {}  # delegation_id → 위임 생성 시각 (조립 지연 측정)
        self._response_started_at: dict[tuple[str | None, str | None], float] = {}  # (delegation, response) → 시작
        self._response_calls: dict[tuple[str | None, str | None], list[str]] = {}  # 로그용 함수 호출 이름

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def start(self) -> LiveStarted:
        """연결하고 ``session.start`` 를 보낸 뒤 ``session.started`` 까지 기다린다.

        연결·수신 실패와 타임아웃은 ``_START_ATTEMPTS`` 번까지 다시 시도한다(인사 WAV 가 나가는 동안이라
        사용자는 차이를 못 느낀다). 서버가 설정을 거부한 경우("rejected")는 다시 해도 같으므로 바로 던진다.

        Raises:
            RuntimeError: 연결·시작 실패, 서버 오류, 타임아웃.
        """
        for attempt in range(1, self._START_ATTEMPTS + 1):
            try:
                return self._start_once()
            except RuntimeError as exc:
                if "rejected" in str(exc) or attempt == self._START_ATTEMPTS:
                    raise
                logger.warning("GPT-Live start attempt %d/%d failed: %s", attempt, self._START_ATTEMPTS, exc)
                time.sleep(self._START_RETRY_DELAY_SEC)
        raise AssertionError("unreachable")

    def _start_once(self) -> LiveStarted:
        t0 = time.monotonic()
        try:
            self._conn = self._connection_factory()
            self._conn.session.start(session=self._config.to_session_dict())
        except Exception as exc:
            trace.record_call("gpt_live", "session_start", self._config.model, _ms(t0), status="error")
            raise RuntimeError(f"GPT-Live connect failed: {exc}") from exc

        deadline = time.monotonic() + self._START_TIMEOUT_SEC
        while True:
            if time.monotonic() > deadline:
                self._abort()
                trace.record_call("gpt_live", "session_start", self._config.model, _ms(t0), status="timeout")
                raise RuntimeError("GPT-Live session.started timeout")
            try:
                event = self._conn.recv()
            except Exception as exc:
                self._abort()
                raise RuntimeError(f"GPT-Live recv failed before start: {exc}") from exc
            etype = getattr(event, "type", None)
            if etype == "error":
                self._abort()
                err = getattr(event, "error", None)
                trace.record_call("gpt_live", "session_start", self._config.model, _ms(t0), status="error")
                raise RuntimeError(f"GPT-Live session rejected: {getattr(err, 'message', err)}")
            if etype == "session.started":
                break
            logger.debug("Ignoring pre-start event %s", etype)

        self._session_id = event.session.id
        started = LiveStarted(session_id=event.session.id, expires_at=float(event.session.expires_at))
        trace.record_call("gpt_live", "session_start", self._config.model, _ms(t0), status="ok")

        self._grouper = self._make_grouper()
        self._stop.clear()
        self._receiver = threading.Thread(target=self._receive_loop, daemon=True, name="gpt-live-recv")
        self._receiver.start()
        logger.debug("GPT-Live session started: %s", started.session_id)
        return started

    def close(self, *, graceful: bool = True) -> None:
        """세션을 닫는다.

        graceful 이면 ``session.close`` 를 보내고 소켓을 닫는다. ``session.closed`` 는 기다리지 않는다 —
        위임이 미완이면 서버가 약 9초 뒤에 보내는데(실측), 과금은 요청 시점에 멈추므로 기다릴 이유가 없다.
        """
        if self._conn is None:
            return
        if graceful:
            try:
                self._conn.session.close(event_id="ray_close")
            except Exception:
                logger.debug("session.close send failed (suppressed)", exc_info=True)
        self._abort()
        logger.debug("GPT-Live session closed")

    def _abort(self) -> None:
        self._stop.set()
        if self._grouper is not None:
            try:
                self._grouper.close()
            except Exception:
                logger.debug("grouper close failed (suppressed)", exc_info=True)
            self._grouper = None
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                logger.debug("connection close failed (suppressed)", exc_info=True)
            self._conn = None
        if self._receiver is not None and self._receiver is not threading.current_thread():
            self._receiver.join(timeout=self._RECV_JOIN_TIMEOUT_SEC)
            self._receiver = None

    # ------------------------------------------------------------------
    # Client → server
    # ------------------------------------------------------------------

    def send_audio(self, pcm: bytes) -> None:
        """마이크 오디오 한 조각(``sample_rate`` mono 16-bit PCM)을 보낸다. ack 없음."""
        self._guard()
        self._send(lambda c: c.session.input_audio.append(audio=base64.b64encode(pcm).decode("ascii")))

    def mute_input(self) -> None:
        """서버가 이후 입력 오디오를 무시하게 한다. 마이크 자체와는 무관하다."""
        self._guard()
        self._send(lambda c: c.session.input_audio.mute(event_id="ray_mute"))

    def append_thinking(self, content: str) -> None:
        """조용한 사실 컨텍스트. 발화 중에도 끊지 않고 반영된다. 500 토큰 이하."""
        self._append("thinking", content)

    def append_instructions(self, content: str) -> None:
        """지시 추가. 모델이 말하는 중이면 발화가 끊기므로 조용할 때 부를 것. 500 토큰 이하."""
        self._append("instructions", content)

    def append_commentary(self, content: str) -> None:
        """모델이 말로 옮길 정보. 패러프레이즈된다. 500 토큰 이하."""
        self._append("commentary", content)

    def _append(self, kind: str, content: str) -> None:
        self._guard()
        self._send(lambda c: getattr(c.session, kind).append(content=content, delegation_id=None))

    def update_backend(self, instructions: str, tools: tuple[dict[str, Any], ...] | None = None) -> None:
        """responses 백엔드의 지시문·툴을 바꾼다 (이후 위임부터 적용). 세션 중 바꿀 수 있는 유일한 설정.

        Args:
            instructions: 새 백엔드 지시문.
            tools: 새 툴 집합. None 이면 세션 시작 설정의 툴(web_search 포함)을 그대로 쓴다 — 지시문만 바꿀 때.
        """
        self._guard()
        if tools is None:
            tools = ({"type": "web_search"}, *self._config.tools)
        delegation = {
            "type": "responses",
            "responses": {
                "instructions": instructions,
                "tools": list(tools),
                "reasoning": {"effort": self._config.backend_reasoning},
            },
        }
        self._send(lambda c: c.session.update(session={"delegation": delegation}, event_id="ray_backend_update"))

    def restore_backend(self) -> None:
        """백엔드 지시문·툴을 세션 시작 설정으로 되돌린다."""
        self.update_backend(self._config.backend_instructions)

    def submit_function_output(self, call_id: str, output: str) -> None:
        """백엔드가 요청한 함수의 결과를 넣는다. 모든 결과를 넣은 뒤 :meth:`continue_response`."""
        self._guard()
        self._send(
            lambda c: c.response.item.create(
                item={"type": "function_call_output", "call_id": call_id, "output": output},
                event_id=f"tool_{call_id}",
            )
        )

    def continue_response(self) -> None:
        """함수 결과를 반영해 백엔드 응답을 이어가게 한다."""
        self._guard()
        self._send(lambda c: c.response.create(event_id="ray_continue"))

    # ------------------------------------------------------------------
    # Server → client
    # ------------------------------------------------------------------

    def poll_event(self) -> LiveEvent | None:
        """다음 이벤트를 꺼낸다. 없으면 None. 수신 스레드 오류가 있으면 RuntimeError."""
        self._check_error()
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _default_connection_factory() -> Any:
        import openai

        return openai.OpenAI().live.connect(max_retries=0).enter()

    def _make_grouper(self) -> Any:
        from openai.lib.live import TranscriptGrouper

        grouper = TranscriptGrouper()
        grouper.on("segment.updated", self._on_segment_updated)
        grouper.on("segment.closed", self._on_segment_closed)
        return grouper

    def _on_segment_updated(self, seg: Any) -> None:
        self._events.put(LiveTranscript(seg.id, seg.speaker, seg.text, seg.start_ms, seg.end_ms, closed=False))

    def _on_segment_closed(self, ev: Any) -> None:
        seg = ev.segment
        self._events.put(
            LiveTranscript(seg.id, seg.speaker, seg.text, seg.start_ms, seg.end_ms, closed=True, reason=ev.reason)
        )

    def _guard(self) -> None:
        self._check_error()
        if self._conn is None:
            raise RuntimeError("GPT-Live session not started")

    def _send(self, fn: Callable[[Any], None]) -> None:
        try:
            fn(self._conn)
        except Exception as exc:
            raise RuntimeError(f"GPT-Live send failed: {exc}") from exc

    def _check_error(self) -> None:
        with self._lock:
            if self._error is not None:
                err, self._error = self._error, None
                raise err

    def _receive_loop(self) -> None:
        conn = self._conn
        while not self._stop.is_set():
            try:
                event = conn.recv()
            except Exception as exc:
                if not self._stop.is_set() and not self._closed_seen.is_set():
                    logger.warning("GPT-Live connection lost: %s", exc)
                    self._events.put(LiveClosed(reason="connection_lost"))
                return
            try:
                self._dispatch(event)
            except Exception:
                logger.warning("GPT-Live event handling failed: %s", getattr(event, "type", event), exc_info=True)

    def _dispatch(self, event: Any) -> None:
        t = getattr(event, "type", None)
        if t == "session.output_audio.delta":
            self._events.put(LiveAudio(base64.b64decode(event.delta)))
        elif t in ("session.input_transcript.delta", "session.output_transcript.delta"):
            if self._grouper is not None:
                self._grouper.push(event)
        elif t == "response.event":
            self._dispatch_backend(event)
        elif t == "session.usage.updated":
            cw = getattr(event, "context_window", None)
            self._events.put(LiveUsage(event.usage.seconds, cw.usage_ratio if cw is not None else None))
        elif t == "session.closed":
            self._closed_seen.set()
            if self._grouper is not None:
                self._grouper.push(event)  # 열린 세그먼트를 닫는다
            self._events.put(LiveClosed(reason=event.reason))
        elif t == "error":
            err = getattr(event, "error", None)
            self._events.put(LiveError(str(getattr(err, "code", "")), str(getattr(err, "message", err))))
        elif t == "session.delegation.created":
            d = getattr(event, "delegation", event)
            delegation_id = getattr(d, "id", None)
            client_event_id = getattr(event, "client_event_id", None)
            self._delegation_created_at[delegation_id] = time.monotonic()
            self._events.put(LiveDelegationCreated(delegation_id, client_event_id))
            logger.info(
                "Delegation created (id=%s, offset=%sms%s)",
                delegation_id or "?",
                getattr(event, "offset_ms", "?"),
                f", by {client_event_id}" if client_event_id else "",
            )
        # 나머지(appended ack, muted, info …)는 상태 변화 없음 — 로그만
        else:
            logger.debug("GPT-Live event %s", t)

    def _dispatch_backend(self, envelope: Any) -> None:
        inner: dict[str, Any] = envelope.event
        it = inner.get("type")
        delegation_id = envelope.delegation_id
        response = inner.get("response") or {}
        response_id = inner.get("response_id") or response.get("id") or self._active_responses.get(delegation_id)
        key = (delegation_id, response_id)
        if it == "response.created" and response_id:
            self._active_responses[delegation_id] = response_id
            self._response_started_at[key] = time.monotonic()
            self._response_calls[key] = []
        elif it == "response.output_item.done":
            item = inner.get("item") or {}
            if item.get("type") == "message":
                parts = item.get("content") or []
                text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict)).strip()
                logger.info("Backend reply: %s", repr(text[:200]) if text else "(empty)")
            if item.get("type") == "function_call":
                self._response_calls.setdefault(key, []).append(str(item.get("name", "")))
                self._events.put(
                    LiveFunctionCall(
                        call_id=str(item.get("call_id", "")),
                        name=str(item.get("name", "")),
                        arguments=str(item.get("arguments") or "{}"),
                        delegation_id=envelope.delegation_id,
                        response_id=response_id,
                    )
                )
        elif it in ("response.completed", "response.failed", "response.incomplete"):
            if it != "response.completed":
                logger.warning("Backend response %s", it)
            if self._active_responses.get(delegation_id) == response_id:
                self._active_responses.pop(delegation_id, None)
            self._log_backend_timing(delegation_id, key, it.removeprefix("response."))
            self._events.put(LiveResponseDone(delegation_id, response_id, it.removeprefix("response.")))

    def _log_backend_timing(self, delegation_id: str | None, key: tuple[str | None, str | None], status: str) -> None:
        """백엔드 응답 1건의 소요 시간 — 로그 한 줄 + 호출 스토어. 위임 생성 → 응답 시작(서버 조립) 간격도 함께.

        응답 id 도 남긴다 — 백엔드 응답은 OpenAI 에 저장돼 ``responses.retrieve`` / ``responses.input_items.list`` 로
        백엔드가 실제로 받은 지시문·입력을 나중에 꺼내 볼 수 있다.
        """
        now = time.monotonic()
        started = self._response_started_at.pop(key, None)
        calls = self._response_calls.pop(key, [])
        if started is None:
            return
        response_id = key[1]
        elapsed_ms = (now - started) * 1000.0
        created = self._delegation_created_at.pop(delegation_id, None)  # 후속 응답(툴 결과 뒤)엔 없다
        setup = f", setup {(started - created) * 1000.0:.0f}ms" if created is not None else ""
        logger.info(
            "Backend response %.2fs (%s%s, calls=%s) %s", elapsed_ms / 1000.0, status, setup, calls or "-", response_id
        )
        setup_ms = round((started - created) * 1000.0) if created else None
        trace.record_call(
            "gpt_live",
            "backend_response",
            self._config.backend_model or "client",
            elapsed_ms,
            status=status,
            metadata=json.dumps({"calls": calls, "setup_ms": setup_ms, "response_id": response_id}),
        )


def _ms(t0: float) -> float:
    return (time.monotonic() - t0) * 1000.0
