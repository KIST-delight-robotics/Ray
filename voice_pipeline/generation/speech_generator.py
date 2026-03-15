"""SpeechGenerator: background ContextBuilder → LLM → TTS pipeline."""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from voice_pipeline.core.config import SpeechGeneratorConfig
from voice_pipeline.core.interfaces import ILLM, ITTS, IContextBuilder, ISpeechGenerator
from voice_pipeline.core.types import GeneratorState, ResponseData

logger = logging.getLogger("voice_pipeline.generation")


class SpeechGenerator(ISpeechGenerator):
    """Chains ContextBuilder → LLM → TTS in a background thread.

    Each prepare() submits a pipeline run. Audio chunks are streamed
    via poll_audio(). Run-ID guards prevent stale runs from writing state.
    """

    def __init__(
        self,
        context_builder: IContextBuilder,
        llm: ILLM,
        tts: ITTS,
        config: SpeechGeneratorConfig | None = None,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._context_builder = context_builder
        self._llm = llm
        self._tts = tts
        self._config = config or SpeechGeneratorConfig()

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

    def _run_pipeline(
        self,
        current_text: str,
        run_id: int,
        cancel_event: threading.Event,
        audio_queue: queue.Queue[bytes],
    ) -> None:
        try:
            t0 = time.monotonic()

            # 1. Build context
            if cancel_event.is_set():
                return
            messages = self._context_builder.build(current_text)

            # 2. Generate LLM text
            if cancel_event.is_set():
                return
            llm_iter = self._llm.generate(messages)
            text_chunks: list[str] = []
            try:
                for chunk in llm_iter:
                    if cancel_event.is_set():
                        if hasattr(llm_iter, "close"):
                            llm_iter.close()
                        return
                    text_chunks.append(chunk)
            except Exception:
                if hasattr(llm_iter, "close"):
                    with contextlib.suppress(Exception):
                        llm_iter.close()
                raise

            full_text = "".join(text_chunks)
            t_llm = time.monotonic()
            logger.info("LLM done (%.1fs) [run=%d]: %r", t_llm - t0, run_id, full_text)

            # 3. Guard: empty text
            if not full_text.strip():
                with self._lock:
                    if run_id == self._run_id:
                        self._state = GeneratorState.FAILED
                return

            # 4. Store text
            with self._lock:
                if run_id != self._run_id:
                    return
                self._text = full_text

            # 5. TTS synthesis
            if cancel_event.is_set():
                return
            tts_stream = self._tts.synthesize(full_text)
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
                t_tts - t_llm, audio_sec, run_id,
            )

            # 6. Build ResponseData
            try:
                timestamps = list(tts_stream.timestamps)
            except Exception:
                logger.debug("Timestamp retrieval failed, using empty list", exc_info=True)
                timestamps = []
            response_data = ResponseData(
                text=full_text,
                audio=bytes(total_audio),
                timestamps=timestamps,
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
