"""Shared helpers for the memory benchmark harness: run-dir layout, config, embedder lock."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

from voice_pipeline.core.interfaces import IEmbedder

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
