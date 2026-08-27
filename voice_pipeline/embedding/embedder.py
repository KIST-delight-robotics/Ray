"""Embedding provider implementations.

Converts text to dense vectors for semantic search.
Shared across modules (memory, similarity, etc.).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np

from voice_pipeline.core.exceptions import ConfigurationError
from voice_pipeline.core.interfaces import IEmbedder

logger = logging.getLogger("voice_pipeline.embedding")


class SentenceTransformerEmbedder(IEmbedder):
    """Embedding via sentence-transformers (local model).

    Supports both PyTorch and ONNX Runtime backends.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        *,
        use_onnx: bool = False,
        expected_dimension: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        local_files_only: bool = False,
    ) -> None:
        """Load a sentence-transformers embedding model.

        Args:
            model: 로드할 sentence-transformers 모델 이름. 기본 ``all-MiniLM-L6-v2``
                (384차원, 다국어 기본 성능 양호).
            use_onnx: ``True``면 ONNX Runtime 백엔드 (CPU inference 가속).
                ``False``면 PyTorch 기본 경로.
            expected_dimension: 지정 시 로드된 모델 차원이 일치하지 않으면
                ``ConfigurationError``. 검증 생략하려면 ``None``.
            model_kwargs: ``SentenceTransformer`` 생성자에 전달할 추가 인자.
                예: ``{"file_name": "onnx/model_qint8_arm64.onnx"}``.
            local_files_only: ``True``면 HF 허브에 접속하지 않고 로컬 캐시만 사용.
                부팅 시 네트워크 의존을 없애고 로드가 ~3.4s 빨라진다(허브 메타데이터
                왕복 생략). 캐시가 없는 새 기기에서는 실패하므로 첫 1회는 ``False``로
                로드해 캐시를 만들어야 한다. 환경변수 ``HF_HUB_OFFLINE=1``은 대안이
                아니다 — sentence-transformers가 ``file_name``을 명시해도 허브 트리
                조회를 시도해 예외로 죽는다.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("sentence-transformers is required for local embeddings. Install with: uv sync") from exc
        backend = "onnx" if use_onnx else "torch"
        self._model = SentenceTransformer(
            model,
            backend=backend,
            model_kwargs=model_kwargs or {},
            local_files_only=local_files_only,
        )
        actual_dim = self._model.get_sentence_embedding_dimension()
        if expected_dimension is not None and actual_dim != expected_dimension:
            raise ConfigurationError(
                f"Embedding model dimension ({actual_dim}) does not match expected_dimension ({expected_dimension})"
            )
        self._dimension = actual_dim
        self._model_name = model
        logger.info(
            "Loaded embedding model: %s (backend=%s, dim=%d)",
            model,
            backend,
            self._dimension,
        )

    def embed(self, text: str) -> np.ndarray:
        vec = self._model.encode(text, show_progress_bar=False)
        return np.asarray(vec, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name


class OpenAIEmbedder(IEmbedder):
    """Embedding via the OpenAI embeddings API.

    If *dimension* is not provided, it is auto-detected from the first
    embed call.  Accessing :attr:`dimension` before any embed raises
    ``RuntimeError``.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        dimension: int | None = None,
    ) -> None:
        """Prepare an OpenAI embeddings client.

        Args:
            model: OpenAI embeddings 모델 이름.
            dimension: 임베딩 벡터 차원. ``None``이면 첫 ``embed`` 호출에서
                자동 감지. ``dimension`` property는 auto-detect 전 접근 시
                ``RuntimeError``.
        """
        try:
            import openai
        except ImportError as exc:
            raise ImportError("openai is required for API embeddings.") from exc
        self._client = openai.OpenAI()
        self._model = model
        self._dimension: int | None = dimension

    def embed(self, text: str) -> np.ndarray:
        response = self._client.embeddings.create(input=[text], model=self._model)
        vec = np.asarray(response.data[0].embedding, dtype=np.float32)
        if self._dimension is None:
            self._dimension = vec.shape[0]
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(input=texts, model=self._model)
        vecs = np.asarray([d.embedding for d in response.data], dtype=np.float32)
        if self._dimension is None and vecs.ndim == 2 and vecs.shape[0] > 0:
            self._dimension = vecs.shape[1]
        return vecs

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("Dimension unknown — call embed() first or provide dimension at construction")
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model


_DEFAULT_MODEL_KWARGS: dict[str, Any] = {"file_name": "onnx/model_qint8_arm64.onnx"}


def create_embedder(
    model: str = "all-MiniLM-L6-v2",
    backend: Literal["local", "api"] = "local",
    *,
    use_onnx: bool = True,
    expected_dimension: int | None = None,
    model_kwargs: dict[str, Any] | None = _DEFAULT_MODEL_KWARGS,
    local_files_only: bool = False,
) -> IEmbedder:
    """Factory: create an IEmbedder instance.

    Args:
        model: 모델 이름. ``"local"`` backend는 sentence-transformers 모델명
            (기본 ``all-MiniLM-L6-v2``, 384차원), ``"api"`` backend는 OpenAI
            embeddings 모델명.
        backend: ``"local"``이면 sentence-transformers, ``"api"``이면 OpenAI.
        use_onnx: ``"local"`` backend에서 ONNX Runtime 사용 여부 (CPU inference 가속).
        expected_dimension: 지정 시 모델 차원과 일치 여부 검증. ``None``이면 검증 생략.
            production wiring은 호출부가 명시 전달. ``"api"`` backend의 auto-detect를
            쓰려면 ``None``.
        model_kwargs: ``SentenceTransformer`` 생성자에 전달할 추가 인자.
            ``"local"`` backend에서만 사용.
        local_files_only: ``"local"`` backend에서 HF 허브 접속 없이 캐시만 사용.
            production wiring은 ``True`` (부팅 시 네트워크 의존 제거). 자세한 제약은
            ``SentenceTransformerEmbedder`` 참고.

    Returns:
        Configured IEmbedder instance.
    """
    if backend == "local":
        return SentenceTransformerEmbedder(
            model,
            use_onnx=use_onnx,
            expected_dimension=expected_dimension,
            model_kwargs=model_kwargs,
            local_files_only=local_files_only,
        )
    elif backend == "api":
        return OpenAIEmbedder(model, dimension=expected_dimension)
    else:
        raise ValueError(f"Unknown embedding backend: {backend!r}")
