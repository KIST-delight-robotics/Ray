"""SpeechGenerator: background ContextBuilder → LLM → TTS pipeline."""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from voice_pipeline.context.context_builder import ContextBuilder
from voice_pipeline.context.formatters import parse_citation_tag, strip_urls
from voice_pipeline.core.interfaces import (
    ILLM,
    ITTS,
    IConversationHistory,
    IHistorySummarizer,
    IMemoryRetriever,
    IMemoryStorage,
    ISpeechGenerator,
    IStorageBackend,
)
from voice_pipeline.core.types import (
    GeneratorState,
    LLMMetrics,
    LLMStream,
    PipelineTrace,
    ResponseData,
    TokenCounter,
    TTSStream,
    WordTimestamp,
)

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

    MAX_WORKERS = 2  # 백그라운드 파이프라인 스레드 풀 크기 — 취소된 run이 API에 blocking되어도 새 prepare 즉시 시작
    _PIPELINE_MODE: Literal["full", "sentence"] = (
        "sentence"  # TTS 파이프라인 모드 — full: LLM 완성 후 TTS, sentence: 문장별 스트리밍
    )
    _QUERY_CONTEXT_TURNS = 3  # 메모리 검색 query에 포함할 최근 history turn 수
    _MIN_FLUSH_WORDS = 4  # sentence 모드에서 TTS flush 전 최소 단어 수 (짧은 감탄사를 다음 문장과 합침)
    _TTS_EXECUTOR_WORKERS = 2  # sentence 모드 TTS 동시 합성 워커 수 — 문장간 gap 최소화
    _CONSUMER_JOIN_TIMEOUT_SEC = 120.0  # sentence consumer 정상 완료 · 개별 문장 TTS future 대기 상한 (초)
    _CANCEL_POLL_INTERVAL_SEC = 0.1  # consumer 스레드 cancel_event 재확인 주기 (초)

    def __init__(
        self,
        llm: ILLM,
        tts: ITTS,
        history: IConversationHistory,
        token_counter: TokenCounter,
        system_prompt: str,
        executor: ThreadPoolExecutor | None = None,
        *,
        memory_storage: IMemoryStorage | None = None,
        retriever: IMemoryRetriever | None = None,
        session_id: str | None = None,
        summarizer: IHistorySummarizer | None = None,
        history_backend: IStorageBackend | None = None,
    ) -> None:
        """Initialize the SpeechGenerator.

        Args:
            llm: LLM 인터페이스.
            tts: TTS 인터페이스.
            history: 세션 대화 이력 (ContextBuilder 및 retriever query에 공유).
            token_counter: 토큰 카운터 콜러블.
            system_prompt: LLM 시스템 프롬프트.
            executor: 백그라운드 파이프라인 ThreadPoolExecutor. ``None``이면
                내부 생성(``MAX_WORKERS`` 기반). 외부 주입 시 shutdown()은
                executor를 닫지 않음.
            memory_storage: 메모리 스토리지 — profiles/summaries 로딩 및
                retriever에 전달. ``None``이면 메모리 사용 안 함.
            retriever: 메모리 retriever. ``None``이면 메모리 검색 안 함.
            session_id: 현재 세션 ID. context 로딩 및 retriever 제외용.
            summarizer: 세션 내 히스토리 롤링 요약기. ``None``이면 요약 없이
                오래된 턴 drop만으로 히스토리 예산을 지킨다.
            history_backend: 직전 세션 이월(carryover) 로딩용 히스토리 백엔드.
                ``None``이면 이월 없음.
        """
        self._context_builder = ContextBuilder(
            history,
            system_prompt,
            token_counter,
            memory_storage=memory_storage,
            session_id=session_id,
            summarizer=summarizer,
            history_backend=history_backend,
        )
        self._llm = llm
        self._tts = tts

        # Memory integration (all optional)
        self._retriever = retriever
        self._history = history
        self._exclude_session_ids = self._context_builder.exclude_session_ids

        self._lock = threading.Lock()
        self._state = GeneratorState.IDLE
        self._run_id = 0
        self._cancel_event = threading.Event()
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(max_workers=self.MAX_WORKERS)
        self._input_text = ""
        self._text = ""
        self._audio_queue: queue.Queue[bytes] = queue.Queue()
        self._response_data: ResponseData | None = None
        self._stream_done = False
        self._trace: PipelineTrace | None = None
        self._memory_results: list[MemoryReadResult | None] = []

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
    def memory_results(self) -> list[MemoryReadResult | None]:
        """Accumulated MemoryReadResults from pipeline runs within this session."""
        return self._memory_results

    @property
    def input_text(self) -> str:
        with self._lock:
            return self._input_text

    @property
    def trace(self) -> PipelineTrace | None:
        with self._lock:
            return self._trace

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
            self._trace = PipelineTrace(
                run_id=run_id,
                pipeline_mode=self._PIPELINE_MODE,
                created_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
                prepare_ts=time.monotonic(),
            )

        logger.debug("prepare(%r) → PREPARING [run=%d]", current_text[:60], run_id)
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
            self._trace = None

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
            self._trace = None
            self._memory_results = []

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
        if self._PIPELINE_MODE == "sentence":
            self._run_pipeline_sentence(current_text, run_id, cancel_event, audio_queue)
        else:
            self._run_pipeline_full(current_text, run_id, cancel_event, audio_queue)

    def _run_pipeline_full(
        self,
        current_text: str,
        run_id: int,
        cancel_event: threading.Event,
        audio_queue: queue.Queue[bytes],
    ) -> None:
        try:
            t0 = time.monotonic()
            trace = self._trace  # snapshot ref — safe under GIL

            if trace is not None:
                trace.pipeline_start_ts = t0

            # 1. Memory retrieval
            if cancel_event.is_set():
                return
            memory_result = self._retrieve_memories(current_text)
            self._memory_results.append(memory_result)

            if trace is not None:
                trace.memory_done_ts = time.monotonic()

            # 2. Build context
            if cancel_event.is_set():
                return
            messages = self._context_builder.build(current_text, memory_result=memory_result)

            if trace is not None:
                trace.context_done_ts = time.monotonic()

            # 3. Generate LLM text
            if cancel_event.is_set():
                return

            if trace is not None:
                trace.llm_start_ts = time.monotonic()

            llm_stream: LLMStream = self._llm.generate(messages)
            text_chunks: list[str] = []
            llm_first = True
            try:
                for chunk in llm_stream:
                    if cancel_event.is_set():
                        llm_stream.close()
                        return
                    if llm_first and trace is not None:
                        trace.llm_first_token_ts = time.monotonic()
                        llm_first = False
                    text_chunks.append(chunk)
            except Exception:
                with contextlib.suppress(Exception):
                    llm_stream.close()
                raise

            full_text = "".join(text_chunks)
            t_llm = time.monotonic()

            if trace is not None:
                trace.llm_done_ts = t_llm

            logger.debug("LLM done (%.1fs) [run=%d]: %r", t_llm - t0, run_id, full_text)

            # 3a. Collect LLM metrics and build turn_items
            metrics_list: list[LLMMetrics] = []
            llm_result = llm_stream.result
            if llm_result.metrics is not None:
                metrics_list.append(llm_result.metrics)
                if trace is not None:
                    trace.llm_ttft_ms = float(llm_result.metrics.ttft_ms)

            # 4. Strip citation tag + URLs
            clean_text, cited_indices = parse_citation_tag(full_text)
            clean_text = strip_urls(clean_text)

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

            if trace is not None:
                trace.tts_start_ts = time.monotonic()

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
                        if trace is not None:
                            trace.tts_first_chunk_ts = time.monotonic()
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

            if trace is not None:
                trace.tts_done_ts = t_tts

            audio_sec = len(total_audio) / (self._tts.output_sample_rate * 2)
            logger.debug(
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

    # -- Sentence-streaming pipeline -----------------------------------------

    # Queue item: (sentence_text, future) or None (sentinel).
    _SentenceItem = tuple[str, "Future[TTSStream]"]

    def _run_pipeline_sentence(
        self,
        current_text: str,
        run_id: int,
        cancel_event: threading.Event,
        audio_queue: queue.Queue[bytes],
    ) -> None:
        from voice_pipeline.generation.sentence_detector import SentenceDetector

        tts_executor: ThreadPoolExecutor | None = None
        consumer_thread: threading.Thread | None = None
        future_queue: queue.Queue[SpeechGenerator._SentenceItem | None] = queue.Queue()

        try:
            t0 = time.monotonic()
            trace = self._trace

            if trace is not None:
                trace.pipeline_start_ts = t0

            # 1. Memory retrieval
            if cancel_event.is_set():
                return
            memory_result = self._retrieve_memories(current_text)
            self._memory_results.append(memory_result)

            if trace is not None:
                trace.memory_done_ts = time.monotonic()

            # 2. Build context
            if cancel_event.is_set():
                return
            messages = self._context_builder.build(current_text, memory_result=memory_result)

            if trace is not None:
                trace.context_done_ts = time.monotonic()

            # 3. LLM stream → sentence detection → TTS submission
            if cancel_event.is_set():
                return

            if trace is not None:
                trace.llm_start_ts = time.monotonic()

            llm_stream: LLMStream = self._llm.generate(messages)

            detector = SentenceDetector(min_flush_words=self._MIN_FLUSH_WORDS)
            tts_executor = ThreadPoolExecutor(max_workers=self._TTS_EXECUTOR_WORKERS)

            # Shared state between producer (this thread) and consumer.
            # Consumer writes; producer reads only after consumer_thread.join().
            accumulated_text: list[str] = []
            total_audio = bytearray()
            all_timestamps: list[WordTimestamp] = []
            consumer_error: list[Exception] = []

            consumer_thread = threading.Thread(
                target=self._sentence_tts_consumer,
                args=(
                    future_queue,
                    audio_queue,
                    cancel_event,
                    run_id,
                    accumulated_text,
                    total_audio,
                    all_timestamps,
                    consumer_error,
                ),
                daemon=True,
            )
            consumer_thread.start()

            # --- Producer: iterate LLM stream, detect sentences, submit TTS ---
            text_chunks: list[str] = []
            llm_first = True
            try:
                for chunk in llm_stream:
                    if cancel_event.is_set():
                        llm_stream.close()
                        return
                    if llm_first and trace is not None:
                        trace.llm_first_token_ts = time.monotonic()
                        llm_first = False
                    text_chunks.append(chunk)
                    for sentence in detector.feed(chunk):
                        if cancel_event.is_set():
                            llm_stream.close()
                            return
                        sentence = strip_urls(sentence)
                        if not sentence:
                            continue
                        future = tts_executor.submit(self._tts.synthesize, sentence)
                        future_queue.put((sentence, future))
            except Exception:
                with contextlib.suppress(Exception):
                    llm_stream.close()
                raise

            full_text = "".join(text_chunks)
            t_llm = time.monotonic()

            if trace is not None:
                trace.llm_done_ts = t_llm

            logger.debug("LLM done (%.1fs) [run=%d]: %r", t_llm - t0, run_id, full_text)

            # Collect LLM metrics
            metrics_list: list[LLMMetrics] = []
            llm_result = llm_stream.result
            if llm_result.metrics is not None:
                metrics_list.append(llm_result.metrics)
                if trace is not None:
                    trace.llm_ttft_ms = float(llm_result.metrics.ttft_ms)

            # 4. Flush remaining buffer + parse citation tag
            cited_indices: list[int] = []
            remainder = detector.flush()
            if remainder:
                clean_remainder, cited_indices = parse_citation_tag(remainder)
                clean_remainder = strip_urls(clean_remainder).strip()
                if clean_remainder:
                    future = tts_executor.submit(self._tts.synthesize, clean_remainder)
                    future_queue.put((clean_remainder, future))
            else:
                _, cited_indices = parse_citation_tag(full_text)

            # 5. Signal consumer to finish, then wait
            future_queue.put(None)
            consumer_thread.join(timeout=self._CONSUMER_JOIN_TIMEOUT_SEC)
            consumer_thread = None  # prevent finally from joining again

            if consumer_error:
                raise consumer_error[0]

            # 6. Build final clean text
            clean_text = " ".join(accumulated_text)

            # 7. Guard: empty text
            if not clean_text.strip():
                with self._lock:
                    if run_id == self._run_id:
                        self._state = GeneratorState.FAILED
                return

            # 7a. Update citations
            cited_ids = self._resolve_citations(cited_indices, memory_result)

            # 8. Guard: no audio produced
            if not total_audio:
                with self._lock:
                    if run_id == self._run_id:
                        self._state = GeneratorState.FAILED
                return

            t_done = time.monotonic()

            if trace is not None:
                trace.tts_done_ts = t_done

            audio_sec = len(total_audio) / (self._tts.output_sample_rate * 2)
            logger.debug(
                "Sentence pipeline done (%.1fs): %.1fs audio [run=%d]",
                t_done - t0,
                audio_sec,
                run_id,
            )

            # 9. Build ResponseData
            turn_items = [{"role": "assistant", "content": clean_text}]
            response_data = ResponseData(
                text=clean_text,
                audio=bytes(total_audio),
                timestamps=all_timestamps,
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
            logger.warning("Sentence pipeline run failed", exc_info=True)
            with self._lock:
                if run_id == self._run_id:
                    self._state = GeneratorState.FAILED
        finally:
            # Ensure consumer is stopped even on producer error.
            if consumer_thread is not None and consumer_thread.is_alive():
                future_queue.put(None)
                # finally cleanup bound — 정상 경로(_CONSUMER_JOIN_TIMEOUT_SEC)와 의미 다름
                consumer_thread.join(timeout=10.0)
            if tts_executor is not None:
                tts_executor.shutdown(wait=False)

    def _sentence_tts_consumer(
        self,
        future_queue: queue.Queue[_SentenceItem | None],
        audio_queue: queue.Queue[bytes],
        cancel_event: threading.Event,
        run_id: int,
        accumulated_text: list[str],
        total_audio: bytearray,
        all_timestamps: list[WordTimestamp],
        consumer_error: list[Exception],
    ) -> None:
        """Consumer thread: drain TTS futures in order, feed *audio_queue*."""
        first_chunk_overall = True
        first_sentence = True
        audio_offset_bytes = 0
        trace = self._trace

        try:
            while True:
                if cancel_event.is_set():
                    return

                # Block with timeout so we can re-check cancel_event.
                try:
                    item = future_queue.get(timeout=self._CANCEL_POLL_INTERVAL_SEC)
                except queue.Empty:
                    continue

                if item is None:
                    return  # sentinel — producer is done

                sentence_text, future = item

                if first_sentence and trace is not None:
                    trace.tts_start_ts = time.monotonic()
                    first_sentence = False

                try:
                    tts_stream: TTSStream = future.result(timeout=self._CONSUMER_JOIN_TIMEOUT_SEC)
                except Exception as exc:
                    consumer_error.append(exc)
                    return

                accumulated_text.append(sentence_text)
                with self._lock:
                    if run_id != self._run_id:
                        return
                    self._text = " ".join(accumulated_text)

                # Drain this sentence's TTS stream.
                sentence_audio = bytearray()
                try:
                    for chunk in tts_stream:
                        if cancel_event.is_set():
                            tts_stream.close()
                            return

                        if first_chunk_overall:
                            with self._lock:
                                if run_id != self._run_id:
                                    tts_stream.close()
                                    return
                                self._state = GeneratorState.STREAMING
                            if trace is not None:
                                trace.tts_first_chunk_ts = time.monotonic()
                            first_chunk_overall = False

                        audio_queue.put(chunk)
                        sentence_audio.extend(chunk)
                        total_audio.extend(chunk)
                except Exception as exc:
                    with contextlib.suppress(Exception):
                        tts_stream.close()
                    consumer_error.append(exc)
                    return

                # Collect timestamps with offset correction.
                for wt in tts_stream.timestamps:
                    offset_sec = audio_offset_bytes / (self._tts.output_sample_rate * 2)
                    all_timestamps.append(
                        WordTimestamp(
                            word=wt.word,
                            start_sec=wt.start_sec + offset_sec,
                            end_sec=wt.end_sec + offset_sec,
                        )
                    )

                audio_offset_bytes += len(sentence_audio)

        except Exception as exc:
            consumer_error.append(exc)
            return
