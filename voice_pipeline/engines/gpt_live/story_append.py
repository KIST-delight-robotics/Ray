"""대본 직접 주입 준비·전송. 이벤트와 타이머로 수락·최종 완료를 추적한다."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from uuid import uuid4

from voice_pipeline.adapters.gpt_live import (
    GPTLiveSession,
    LiveAppended,
    LiveClosed,
    LiveError,
    LiveFunctionCall,
    LiveResponseDone,
)
from voice_pipeline.engines.gpt_live.instructions import STORY_MODE_INSTRUCTIONS

logger = logging.getLogger("voice_pipeline.story_append")

STORY_APPEND_PREPARATION = STORY_MODE_INSTRUCTIONS


def narration_text(body: str) -> str:
    """명시적 도입부·본문만 추출한다. 제목·곡 프로필·인용형 진행 메모를 제외한다."""
    sections: dict[str, list[str]] = {"도입부": [], "본문": []}
    current = ""
    for line in body.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
        elif current in sections and line.strip() and not line.lstrip().startswith(">"):
            if line.startswith("#"):
                raise ValueError("unexpected heading in narration")
            sections[current].append(line.strip())
    if not all(sections.values()):
        raise ValueError("script requires nonempty 도입부 and 본문 sections")
    return "\n\n".join("\n".join(sections[name]) for name in ("도입부", "본문"))


def script_parts(
    text: str, story_id: str, transfer_id: str, count_tokens: Callable[[str], int], limit: int = 450
) -> list[str]:
    """문장 경계에서 분할한다. 식별자·번호를 포함한 실제 content 전체에 한도를 적용한다."""
    sentences = re.split(r'(?<=[.!?])(?=\s)|(?<=[.!?]["”])(?=\s)', text)
    upper = len(sentences)

    def wrap(part: str, index: int, total: int) -> str:
        return f"STORY_SCRIPT story={story_id} transfer={transfer_id} part={index}/{total}\n{part}"

    chunks: list[str] = []
    pending = ""
    for sentence in sentences:
        candidate = pending + sentence
        if count_tokens(wrap(candidate, upper, upper)) > limit:
            if pending:
                chunks.append(pending)
            pending = sentence
            if count_tokens(wrap(pending, upper, upper)) > limit:
                raise ValueError("one sentence exceeds the append token budget")
        else:
            pending = candidate
    if pending:
        chunks.append(pending)
    if not chunks:
        raise ValueError("empty narration")
    result = [wrap(chunk, i, len(chunks)) for i, chunk in enumerate(chunks, 1)]
    if any(count_tokens(part) > limit for part in result):
        raise ValueError("append envelope exceeds token budget")
    return result


class StoryAppendTransfer:
    """poll 루프에서 handle/tick을 부른다. ack나 백엔드를 기다리며 프레임 처리를 막지 않는다.

    새 인스턴스마다 고유 transfer ID를 쓰므로 취소 후 늦은 ack가 새 이야기를 시작하지 않는다.
    도구를 호출한 응답이 완료되고 출력이 무음일 때 begin을 호출한다.
    """

    def __init__(
        self,
        live: GPTLiveSession,
        call: LiveFunctionCall,
        story_id: str,
        text: str,
        count_tokens: Callable[[str], int],
        *,
        start_kind: str = "commentary",
        timeout: float = 20.0,
        result_callback: Callable[[str], None] | None = None,
    ) -> None:
        if start_kind not in ("commentary", "instructions"):
            raise ValueError("invalid start kind")
        self.live, self.call, self.story_id = live, call, story_id
        self.result_callback = result_callback
        self.transfer_id = uuid4().hex
        self.parts = script_parts(text, story_id, self.transfer_id, count_tokens)
        self.preparation = f"{STORY_APPEND_PREPARATION}\nCurrent transfer: {self.transfer_id}; story: {story_id}."
        self.start_text = (
            f"STORY_START transfer={self.transfer_id} story={story_id}. "
            "Tell the complete supplied story now, through its final song offer."
        )
        if max(count_tokens(self.preparation), count_tokens(self.start_text)) > 450:
            raise ValueError("control instructions exceed token budget")
        self.start_kind, self.timeout = start_kind, timeout
        self.state = "new"
        self.pending_id: str | None = None
        self.pending_kind: str | None = None
        self.deadline = 0.0
        self.next_part = 0
        self.result_submitted = False
        self.failure: str | None = None
        self.timeline: list[dict[str, object]] = []

    def _record(self, event: str, now: float, **values: object) -> None:
        self.timeline.append({"event": event, "time": now, **values})
        logger.info("Story %s transfer=%s %s %s", self.story_id, self.transfer_id, event, values)

    def _send(self, kind: str, content: str, now: float) -> None:
        self.pending_id = f"story_{self.transfer_id}_{len(self.timeline)}"
        self.pending_kind = kind
        self.deadline = now + self.timeout
        getattr(self.live, f"append_{kind}")(content, event_id=self.pending_id)
        self._record("append_sent", now, kind=kind, event_id=self.pending_id)

    def begin(self, now: float) -> None:
        if self.state != "new":
            raise RuntimeError("transfer already started")
        self.state = "preparing"
        self._send("instructions", self.preparation, now)

    def handle(self, event: object, now: float) -> None:
        if self.state in ("new", "started", "failed", "cancelled"):
            return
        if isinstance(event, LiveClosed):
            self.cancel(now, "session closed", submit_error=False)
        elif isinstance(event, LiveError) and event.client_event_id == self.pending_id:
            self.cancel(now, event.message)
        elif isinstance(event, LiveAppended):
            if event.client_event_id != self.pending_id or event.kind != self.pending_kind:
                return
            self._record(
                "append_accepted", now, event_id=event.client_event_id, start_ms=event.start_ms, end_ms=event.end_ms
            )
            self.pending_id = self.pending_kind = None
            if self.state == "starting":
                self.state = "started"
            elif self.next_part < len(self.parts):
                self.state = "injecting"
                self._send("thinking", self.parts[self.next_part], now)
                self.next_part += 1
            else:
                self.state = "backend"
                self.deadline = now + self.timeout
                result = json.dumps({"status": "prepared", "story": self.story_id, "transfer": self.transfer_id})
                if self.result_callback is not None:
                    self.result_callback(result)
                else:
                    self.live.submit_function_output(self.call.call_id, result)
                    self.result_submitted = True
                    self.live.continue_response()
                self._record("script_prepared", now)
        elif isinstance(event, LiveResponseDone) and event.delegation_id == self.call.delegation_id:
            if self.state != "backend" or not self.result_submitted or event.response_id == self.call.response_id:
                return
            self._record(
                "backend_done",
                now,
                response_id=event.response_id,
                status=event.status,
                function_call_ids=event.function_call_ids,
                is_final=event.is_final,
            )
            if event.is_final:
                self.state = "starting"
                self._send(self.start_kind, self.start_text, now)
            elif event.status != "completed":
                self.cancel(now, "backend did not finish successfully")

    def tick(self, now: float) -> None:
        if self.state not in ("new", "started", "failed", "cancelled") and now >= self.deadline:
            self.cancel(now, f"timeout during {self.state}")

    def cancel(self, now: float, reason: str = "cancelled", *, submit_error: bool = True) -> None:
        if self.state in ("failed", "cancelled"):
            return
        self.state = "cancelled" if reason == "cancelled" else "failed"
        self.failure = reason
        self.pending_id = self.pending_kind = None
        self._record(self.state, now, reason=reason)
        # Resolve a pending result even when the socket is already closed.
        if self.result_callback is not None and not self.result_submitted:
            self.result_callback(json.dumps({"error": reason}))
        if submit_error:
            try:
                self.live.append_instructions(
                    f"Transfer {self.transfer_id} for story {self.story_id} is cancelled. "
                    "Do not narrate or use its script. Wait for a new user request."
                )
            finally:
                if self.result_callback is None and not self.result_submitted:
                    self.live.submit_function_output(self.call.call_id, json.dumps({"error": reason}))
                    self.result_submitted = True
                    self.live.continue_response()
