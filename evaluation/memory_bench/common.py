"""Shared helpers for the memory benchmark harness: run-dir layout, config, embedder lock."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np

from voice_pipeline.core.interfaces import ILLM, IEmbedder
from voice_pipeline.core.types import LLMResult, LLMStream

CONFIG_FILENAME = "config.json"
ANSWERS_FILENAME = "answers.jsonl"
SCORES_FILENAME = "scores.json"
DBS_DIRNAME = "dbs"

DEFAULT_WRITER_MODEL = "gpt-4o-mini"  # 프로덕션 MemoryWriter와 동일 (voice_pipeline/__main__.py)
DEFAULT_ANSWER_MODEL = "gpt-4o-mini"  # Mem0 LoCoMo 평가와 비교 가능하도록 동일 모델
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"


def slugify(name: str) -> str:
    """Lowercase a speaker name into a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def db_path(run_dir: Path, sample_id: str, user_speaker: str) -> Path:
    """SQLite path for one (conversation, user-perspective) ingest unit."""
    return run_dir / DBS_DIRNAME / f"{sample_id}_{slugify(user_speaker)}.db"


def session_db_id(sample_id: str, session_index: int) -> str:
    """Memory-storage session ID for a LoCoMo session."""
    return f"{sample_id}_s{session_index:02d}"


def load_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / CONFIG_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def update_config(run_dir: Path, section: str, payload: dict[str, Any]) -> None:
    """Merge a phase config section into the run's ``config.json``."""
    config = load_config(run_dir)
    config[section] = payload
    (run_dir / CONFIG_FILENAME).write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")


def read_answers(run_dir: Path) -> list[dict[str, Any]]:
    """Read all records from ``answers.jsonl`` (empty list if absent)."""
    path = run_dir / ANSWERS_FILENAME
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class JsonlWriter:
    """Append-only JSONL writer safe for concurrent worker threads."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self._path.open("a") as f:
                f.write(line + "\n")


class LockedEmbedder(IEmbedder):
    """Serialize access to a shared embedder across worker threads.

    Local sentence-transformers/ONNX 인스턴스를 스레드마다 새로 만들면 로드
    비용이 크므로 하나를 공유하고 호출만 직렬화한다. 임베딩(수 ms)은 LLM 콜
    (수 초)에 비해 짧아 락 경합은 무시할 수준.
    """

    def __init__(self, inner: IEmbedder) -> None:
        self._inner = inner
        self._lock = threading.Lock()

    def embed(self, text: str) -> np.ndarray:
        with self._lock:
            return self._inner.embed(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            return self._inner.embed_batch(texts)

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def model_name(self) -> str:
        return self._inner.model_name


class UsageTrackingLLM(ILLM):
    """Delegate ILLM that accumulates token usage across calls (thread-safe).

    벤치 각 단계(ingest/answer/score)의 LLM을 감싸 응답 usage(input/output/
    cached 토큰)를 합산한다. 합계는 run config의 해당 단계 섹션에 기록된다.
    스트림을 끝까지 소비한 호출만 집계된다 (벤치는 항상 완주).
    """

    def __init__(self, inner: ILLM) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.missing_usage = 0  # usage가 응답에 없던 호출 수

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMStream:
        """Delegate to the inner LLM and record usage after full iteration."""
        stream = self._inner.generate(messages, tools=tools, response_format=response_format)

        def gen() -> Generator[str, None, None]:
            yield from stream
            self._record(stream.result)

        return LLMStream(gen(), close_fn=stream.close, result_fn=lambda _text: stream.result)

    def _record(self, result: LLMResult) -> None:
        with self._lock:
            self.calls += 1
            metrics = result.metrics
            if metrics is None:
                self.missing_usage += 1
                return
            self.input_tokens += metrics.usage.input_tokens
            self.output_tokens += metrics.usage.output_tokens
            self.cached_tokens += metrics.usage.cached_tokens

    def summary(self) -> dict[str, int]:
        """Accumulated usage totals (config 기록용)."""
        with self._lock:
            return {
                "calls": self.calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cached_tokens": self.cached_tokens,
                "missing_usage": self.missing_usage,
            }
