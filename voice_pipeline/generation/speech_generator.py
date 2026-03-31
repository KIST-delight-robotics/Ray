"""SpeechGenerator: background ContextBuilder → LLM → TTS pipeline."""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from voice_pipeline.context.formatters import parse_citation_tag
from voice_pipeline.core.config import SpeechGeneratorConfig
from voice_pipeline.core.interfaces import (
    ILLM,
    ITTS,
    IContextBuilder,
    IConversationHistory,
    IMemoryRetriever,
    ISpeechGenerator,
)
from voice_pipeline.core.types import GeneratorState, LLMMetrics, LLMStream, ResponseData

if TYPE_CHECKING:
    from voice_pipeline.memory.types import MemoryReadResult

logger = logging.getLogger("voice_pipeline.generation")


class SpeechGenerator(ISpeechGenerator):
    """Chains ContextBuilder → LLM → TTS in a background thread.

    Each prepare() submits a pipeline run. Audio chunks are streamed
    via poll_audio(). Run-ID guards prevent stale runs from writing state.

    When a retriever is provided, the pipeline additionally:
      1. Builds a retrieval query from current text + recent history.
      2. Calls retriever.retrieve() before context assembly.
      3. Parses ``[MEMORIES: ...]`` from LLM output and strips it before TTS.
      4. Calls retriever.update_citations() with cited indices.
    """

    def __init__(
        self,
        context_builder: IContextBuilder,
        llm: ILLM,
        tts: ITTS,
        config: SpeechGeneratorConfig | None = None,
        executor: ThreadPoolExecutor | None = None,
        *,
        retriever: IMemoryRetriever | None = None,
        history: IConversationHistory | None = None,
        exclude_session_ids: set[str] | None = None,
    ) -> None:
        self._context_builder = context_builder
        self._llm = llm
        self._tts = tts
        self._config = config or SpeechGeneratorConfig()

        # Memory integration (all optional)
        self._retriever = retriever
        self._history = history
        self._exclude_session_ids = exclude_session_ids or set()

        self._lock = threading.Lock()
        self._state = GeneratorState.IDLE
        self._run_id = 0
        self._cancel_event = threading.Event()
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(max_workers=self._config.max_workers)
        self._input_text = ""
        self._text = ""
        self._audio_queue: queue.Queue[bytes] = queue.Queue()
        self._response_data: ResponseData | None = None
        self._stream_done = False

    # -- Properties ----------------------------------------------------------

    @property
    def state(self) -> GeneratorState:
        with self._lock:
            return self._state

    @property
    def stream_done(self) -> bool:
        with self._lock:
            return self._stream_done

    @property
    def input_text(self) -> str:
        with self._lock:
            return self._input_text

    # -- Public methods ------------------------------------------------------

    def prepare(self, current_text: str) -> None:
        with self._lock:
            # Cancel previous run
            self._cancel_event.set()

            # New run
            self._run_id += 1
            run_id = self._run_id
            self._cancel_event = threading.Event()
            cancel_event = self._cancel_event
            self._audio_queue = queue.Queue()
            audio_queue = self._audio_queue

            self._state = GeneratorState.PREPARING
            self._input_text = current_text
            self._text = ""
            self._response_data = None
            self._stream_done = False

        logger.info("prepare(%r) → PREPARING [run=%d]", current_text[:60], run_id)
        self._executor.submit(self._run_pipeline, current_text, run_id, cancel_event, audio_queue)

    def cancel(self) -> None:
        with self._lock:
            self._cancel_event.set()
            self._run_id += 1
            logger.info("cancel → IDLE [run=%d]", self._run_id)
            self._state = GeneratorState.IDLE
            self._audio_queue = queue.Queue()
            self._input_text = ""
            self._text = ""
            self._response_data = None
            self._stream_done = False

    def poll_audio(self) -> bytes | None:
        with self._lock:
            try:
                return self._audio_queue.get_nowait()
            except queue.Empty:
                return None

    def get_text(self) -> str:
        with self._lock:
            allowed = (GeneratorState.STREAMING, GeneratorState.IDLE, GeneratorState.FAILED)
            if self._state not in allowed or not self._text:
                raise RuntimeError(f"Text not available in state {self._state.value}")
            return self._text

    def get_response_data(self) -> ResponseData:
        with self._lock:
            if not self._stream_done:
                raise RuntimeError("Stream not done — cannot get response data")
            if self._response_data is None:
                raise RuntimeError("No response data available")
            data = self._response_data
            self._state = GeneratorState.IDLE
            self._input_text = ""
            return data

    def reset(self) -> None:
        """Cancel any running pipeline and reset state for the next session."""
        with self._lock:
            self._cancel_event.set()
            self._run_id += 1
            self._state = GeneratorState.IDLE
            self._audio_queue = queue.Queue()
            self._input_text = ""
            self._text = ""
            self._response_data = None
            self._stream_done = False

    def shutdown(self) -> None:
        """Permanently shut down the executor. Call only at program exit.

        If the executor was injected externally, only cancels in-flight work
        without shutting down the executor (caller owns the lifecycle).
        """
        with self._lock:
            self._cancel_event.set()
        if self._owns_executor:
            self._executor.shutdown(wait=True)

    # -- Background pipeline -------------------------------------------------

    def _build_retriever_query(self, current_text: str) -> str:
        """Concatenate current text with recent history turns for retrieval."""
        if self._history is None:
            return current_text
        turns = self._history.get_turns()
        recent = turns[-self._config.query_context_turns :]
        parts: list[str] = []
        for turn in recent:
            for item in turn.items:
                content = item.get("content", "")
                if content:
                    parts.append(content)
        parts.append(current_text)
        return " ".join(parts)

    def _retrieve_memories(self, current_text: str) -> MemoryReadResult | None:
        """Run memory retrieval. Returns None on error or if no retriever."""
        if self._retriever is None:
            return None
        try:
            query = self._build_retriever_query(current_text)
            return self._retriever.retrieve(query, self._exclude_session_ids)
        except Exception:
            logger.warning("Memory retrieval failed, continuing without", exc_info=True)
            return None

    def _resolve_citations(
        self,
        cited_indices: list[int],
        memory_result: MemoryReadResult | None,
    ) -> list[int]:
        """Update retriever and resolve display indices to DB IDs."""
        if not cited_indices or not memory_result or not self._retriever:
            return []

        try:
            self._retriever.update_citations(cited_indices)
        except Exception:
            logger.warning("Citation update failed", exc_info=True)

        cited_ids: list[int] = []
        for idx in cited_indices:
            db_id = memory_result.index_to_id.get(idx)
            if db_id is not None:
                cited_ids.append(db_id)
        return cited_ids

    def _run_pipeline(
        self,
        current_text: str,
        run_id: int,
        cancel_event: threading.Event,
        audio_queue: queue.Queue[bytes],
    ) -> None:
        try:
            t0 = time.monotonic()

            # 1. Memory retrieval
            if cancel_event.is_set():
                return
            memory_result = self._retrieve_memories(current_text)

            # 2. Build context
            if cancel_event.is_set():
                return
            messages = self._context_builder.build(current_text, memory_result=memory_result)

            # 3. Generate LLM text
            if cancel_event.is_set():
                return
            llm_stream: LLMStream = self._llm.generate(messages)
            text_chunks: list[str] = []
            try:
                for chunk in llm_stream:
                    if cancel_event.is_set():
                        llm_stream.close()
                        return
                    text_chunks.append(chunk)
            except Exception:
                with contextlib.suppress(Exception):
                    llm_stream.close()
                raise

            full_text = "".join(text_chunks)
            t_llm = time.monotonic()
            logger.info("LLM done (%.1fs) [run=%d]: %r", t_llm - t0, run_id, full_text)

            # 3a. Collect LLM metrics and build turn_items
            metrics_list: list[LLMMetrics] = []
            try:
                llm_result = llm_stream.result
                if llm_result.metrics is not None:
                    metrics_list.append(llm_result.metrics)
            except RuntimeError:
                pass  # Stream was closed early, no result available

            # 4. Strip citation tag
            clean_text, cited_indices = parse_citation_tag(full_text)

            # 5. Guard: empty text (before updating citations)
            if not clean_text.strip():
                with self._lock:
                    if run_id == self._run_id:
                        self._state = GeneratorState.FAILED
                return

            # 5a. Update citations (only for non-empty responses)
            cited_ids = self._resolve_citations(cited_indices, memory_result)

            # 6. Store text
            with self._lock:
                if run_id != self._run_id:
                    return
                self._text = clean_text

            # 7. TTS synthesis (using clean text without citation tag)
            if cancel_event.is_set():
                return
            tts_stream = self._tts.synthesize(clean_text)
            first_chunk = True
            total_audio = bytearray()
            try:
                for chunk in tts_stream:
                    if cancel_event.is_set():
                        tts_stream.close()
                        return

                    if first_chunk:
                        with self._lock:
                            if run_id != self._run_id:
                                tts_stream.close()
                                return
                            self._state = GeneratorState.STREAMING
                        first_chunk = False

                    audio_queue.put(chunk)
                    total_audio.extend(chunk)
            except Exception:
                with contextlib.suppress(Exception):
                    tts_stream.close()
                raise

            # Guard: zero chunks from TTS
            if first_chunk:
                with self._lock:
                    if run_id == self._run_id:
                        self._state = GeneratorState.FAILED
                return

            t_tts = time.monotonic()
            audio_sec = len(total_audio) / (24000 * 2)
            logger.info(
                "TTS done (%.1fs): %.1fs audio → STREAMING [run=%d]",
                t_tts - t_llm,
                audio_sec,
                run_id,
            )

            # 8. Build ResponseData
            try:
                timestamps = list(tts_stream.timestamps)
            except Exception:
                logger.debug("Timestamp retrieval failed, using empty list", exc_info=True)
                timestamps = []

            turn_items = [{"role": "assistant", "content": clean_text}]
            response_data = ResponseData(
                text=clean_text,
                audio=bytes(total_audio),
                timestamps=timestamps,
                turn_items=turn_items,
                metrics_list=metrics_list,
                cited_memory_ids=cited_ids,
            )

            with self._lock:
                if run_id != self._run_id:
                    return
                self._response_data = response_data
                self._stream_done = True

        except Exception:
            logger.warning("Pipeline run failed", exc_info=True)
            with self._lock:
                if run_id == self._run_id:
                    self._state = GeneratorState.FAILED
