# Embedding Module

텍스트를 dense vector로 변환하는 `IEmbedder` 구현체와 factory. 메모리 의미 검색(`MemoryRetriever`·`MemoryWriter`)과 턴 분기 유사도 게이트(`TurnDetector`)가 공유 사용.

## Setup

### Local backend (default)

`sentence-transformers`가 `pyproject.toml` 의존성으로 포함. 추가 설치 불필요. 첫 사용 시 HuggingFace에서 모델 파일을 `~/.cache/huggingface/`에 다운로드.

### API backend

OpenAI Embeddings API 사용. `OPENAI_API_KEY` 환경변수 필요.

```bash
export OPENAI_API_KEY=sk-...
```

## `SentenceTransformerEmbedder.__init__` 인자

| 인자 | Default | 의미 |
|---|---|---|
| `model` | `"all-MiniLM-L6-v2"` | 로드할 sentence-transformers 모델 이름 (384차원 기본). |
| `use_onnx` | `False` | ONNX Runtime 백엔드 사용 여부 (CPU 가속). |
| `expected_dimension` | `None` | 지정 시 로드 모델 차원 검증. 불일치하면 `ConfigurationError`. |

## `OpenAIEmbedder.__init__` 인자

| 인자 | Default | 의미 |
|---|---|---|
| `model` | `"text-embedding-3-small"` | OpenAI embeddings 모델 이름. |
| `dimension` | `None` | 벡터 차원. `None`이면 첫 `embed` 호출에서 auto-detect. |

## `create_embedder` 인자

| 인자 | Default | 의미 |
|---|---|---|
| `model` | `"all-MiniLM-L6-v2"` | 모델 이름 (backend별 의미 다름). |
| `backend` | `"local"` | `"local"` → sentence-transformers, `"api"` → OpenAI. |
| `use_onnx` | `False` | `"local"` backend에서 ONNX Runtime 사용 여부. |
| `expected_dimension` | `None` | 지정 시 모델 차원 검증. `"api"` backend의 auto-detect를 쓰려면 `None`. |

## Usage

```python
from voice_pipeline.embedding.embedder import create_embedder

# Default: local backend, PyTorch, all-MiniLM-L6-v2, validation 없음
embedder = create_embedder()

# production wiring: dimension 검증 명시
embedder = create_embedder(expected_dimension=384)

# ONNX 가속
embedder = create_embedder(use_onnx=True, expected_dimension=384)

# API backend (auto-detect dim)
embedder = create_embedder(backend="api")  # text-embedding-3-small, dim 1536

vec = embedder.embed("안녕하세요")           # np.ndarray (384,) float32
vecs = embedder.embed_batch(["a", "b"])    # np.ndarray (2, 384) float32
print(embedder.dimension)                   # 384
```

## Testing

- `tests/embedding/test_embedder.py`: 유닛 테스트 (unknown backend / auto-detect dim / wrong dim 검증). `@pytest.mark.requires_model` 기반 라이브 모델 테스트는 `TestSentenceTransformerEmbedder` 클래스.
- Embedder를 소비하는 통합 테스트는 `tests/memory/conftest.py:shared_embedder` fixture 공유 (session scope) — 모델 로드 1회.
