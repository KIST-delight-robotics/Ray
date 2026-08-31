"""Text-only conversation session — no audio, TTS, or turn-taking.

Chains ConversationHistory → ContextBuilder → LLM → response,
with optional long-term memory retrieval and utterance storage.

Usage::

    session = TextSession(
        llm=llm,
        history=history,
        token_counter=token_counter,
        system_prompt=SYSTEM_PROMPT,
    )
    response = session.send("What is the speed of light?")
    session.close()

For memory-enabled sessions::

    session = TextSession(
        llm=llm,
        history=history,
        token_counter=token_counter,
        system_prompt=SYSTEM_PROMPT,
        memory_storage=memory_storage,
        retriever=retriever,
    )
    response = session.send("What movie did I watch recently?")
    session.close()
"""

from __future__ import annotations

import contextlib
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from voice_pipeline.history import ConversationHistory, SQLiteStorageBackend
from voice_pipeline.memory.retriever import MemoryRetriever
from voice_pipeline.memory.storage import SQLiteMemoryStorage
from voice_pipeline.prompt import ContextBuilder, parse_citation_tag, strip_urls
from voice_pipeline.types import ILLM, LLMMetrics, TokenCounter

if TYPE_CHECKING:
    from voice_pipeline.memory.types import MemoryReadResult

logger = logging.getLogger("voice_pipeline.text_session")


@dataclass
class TextTurnTrace:
    """Timing trace for one TextSession.send() call."""

    memory_ms: float = 0.0
    context_ms: float = 0.0
    llm_ms: float = 0.0
    llm_ttft_ms: float = 0.0
    total_ms: float = 0.0


class TextSession:
    """Text-only conversation session bypassing audio/TTS/turn-taking.

    Each instance represents one conversation session.  Call
    :meth:`send` to exchange messages with the LLM, or :meth:`inject`
    to seed history without generation.
    """

    _QUERY_CONTEXT_TURNS = 3

    def __init__(
        self,
        *,
        llm: ILLM,
        history: ConversationHistory,
        token_counter: TokenCounter,
        system_prompt: str,
        memory_storage: SQLiteMemoryStorage | None = None,
        retriever: MemoryRetriever | None = None,
        session_id: str | None = None,
        load_session_context: bool = True,
        history_backend: SQLiteStorageBackend | None = None,
    ) -> None:
        self._llm = llm
        self._history = history
        self._token_counter = token_counter
        self._retriever = retriever
        self._memory_storage = memory_storage
        self._session_id = session_id or str(uuid.uuid4())
        self._last_metrics: LLMMetrics | None = None
        self._memory_results: list[MemoryReadResult | None] = []
        self._traces: list[TextTurnTrace] = []

        if load_session_context:
            self._context_builder = ContextBuilder(
                history,
                system_prompt,
                token_counter,
                memory_storage=memory_storage,
                session_id=self._session_id,
                history_backend=history_backend,
            )
        else:
            profiles = memory_storage.get_all_profiles() if memory_storage else []
            self._context_builder = ContextBuilder(
                history,
                system_prompt,
                token_counter,
                profiles=profiles,
            )
        self._exclude_session_ids = self._context_builder.exclude_session_ids | {self._session_id}

        self._history.new_session(self._session_id)

    def __enter__(self) -> TextSession:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def last_metrics(self) -> LLMMetrics | None:
        """LLM metrics from the most recent :meth:`send` call."""
        return self._last_metrics

    @property
    def memory_results(self) -> list[MemoryReadResult | None]:
        """MemoryReadResults from each :meth:`send` call, in order."""
        return self._memory_results

    @property
    def traces(self) -> list[TextTurnTrace]:
        """Timing traces from each :meth:`send` call, in order."""
        return self._traces

    def send(self, text: str) -> str:
        """Send user text and return the assistant's response.

        Pipeline: add to history → memory retrieval → context build →
        LLM generate → parse citations → record response.

        Args:
            text: User message text.

        Returns:
            Assistant response text (citations and URLs stripped).

        Raises:
            RuntimeError: If the LLM returns empty text.
        """
        self._last_metrics = None
        trace = TextTurnTrace()

        self._history.add_user_message(text)
        self._store_utterance("user", text)

        t0 = time.monotonic()

        memory_result = self._retrieve_memories(text)
        self._memory_results.append(memory_result)
        t_mem = time.monotonic()
        trace.memory_ms = (t_mem - t0) * 1000

        messages = self._context_builder.build(text, memory_result=memory_result)
        t_ctx = time.monotonic()
        trace.context_ms = (t_ctx - t_mem) * 1000

        llm_stream = self._llm.generate(messages)
        chunks: list[str] = []
        llm_first = True
        try:
            for chunk in llm_stream:
                if llm_first:
                    trace.llm_ttft_ms = (time.monotonic() - t_ctx) * 1000
                    llm_first = False
                chunks.append(chunk)
        except Exception:
            with contextlib.suppress(Exception):
                llm_stream.close()
            raise

        t_llm = time.monotonic()
        trace.llm_ms = (t_llm - t_ctx) * 1000
        trace.total_ms = (t_llm - t0) * 1000
        self._traces.append(trace)

        full_text = "".join(chunks)

        clean_text, cited_indices = parse_citation_tag(full_text)
        clean_text = strip_urls(clean_text).strip()

        if not clean_text:
            raise RuntimeError("LLM returned empty response")

        self._resolve_citations(cited_indices, memory_result)

        metrics = llm_stream.result.metrics
        self._last_metrics = metrics
        self._history.add_assistant_message(clean_text, metrics)
        self._store_utterance("assistant", clean_text)

        return clean_text

    def inject(self, role: str, text: str) -> None:
        """Add a message without LLM generation.

        Use for seeding conversation history (e.g., memory evaluation
        scenarios where both user and assistant turns are scripted).

        Args:
            role: ``"user"`` or ``"assistant"``.
            text: Message text.
        """
        if role == "user":
            self._history.add_user_message(text)
        elif role == "assistant":
            self._history.add_assistant_message(text)
        else:
            raise ValueError(f"Unknown role: {role!r}")
        self._store_utterance(role, text)

    def close(self) -> None:
        """Finalize the session (persists end timestamp)."""
        self._history.save()

    # -- Internal helpers -------------------------------------------------------

    def _store_utterance(self, role: str, text: str) -> None:
        if self._memory_storage is None:
            return
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        token_count = self._token_counter(text)
        self._memory_storage.add_utterance(
            self._session_id,
            role,
            text,
            timestamp,
            token_count,
        )

    def _build_retriever_query(self, current_text: str) -> str:
        turns = self._history.get_turns()
        recent = turns[-self._QUERY_CONTEXT_TURNS :]
        parts: list[str] = []
        for turn in recent:
            for item in turn.items:
                content = item.get("content", "")
                if content:
                    parts.append(content)
        parts.append(current_text)
        return " ".join(parts)

    def _retrieve_memories(self, current_text: str) -> MemoryReadResult | None:
        if self._retriever is None:
            return None
        try:
            query = self._build_retriever_query(current_text)
            return self._retriever.retrieve(query, self._exclude_session_ids)
        except Exception:
            logger.warning("Memory retrieval failed", exc_info=True)
            return None

    def _resolve_citations(
        self,
        cited_indices: list[int],
        memory_result: MemoryReadResult | None,
    ) -> None:
        if not cited_indices or not memory_result or not self._retriever:
            return
        try:
            self._retriever.update_citations(cited_indices)
        except Exception:
            logger.warning("Citation update failed", exc_info=True)
