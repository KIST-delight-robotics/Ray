# Similarity

Semantic text similarity scoring for the voice pipeline. Used by `TurnDetector` to decide whether ASR text has changed enough to warrant a new speculative response (`prepare()`).

## Backends

| Backend | Class | Model example | Latency |
|---------|-------|---------------|---------|
| `local` | `EmbeddingSimilarity` | `all-MiniLM-L6-v2` | ~4ms (CPU) |
| `api` | `EmbeddingSimilarity` | `text-embedding-3-small` | ~200ms (network) |
| `difflib` | `DiffLibSimilarity` | — | <1ms |

Both `local` and `api` backends use `EmbeddingSimilarity`, which wraps an `IEmbedder` instance from the `embedding/` module.

## Configuration

```python
SimilarityConfig(
    backend="local",              # "local", "api", or "difflib"
    model="all-MiniLM-L6-v2",    # sentence-transformers or OpenAI model
    threshold=0.8,                # skip re-prepare if similarity >= this
    use_onnx=False,               # ONNX Runtime backend (local only)
)
```

## Dependencies

- **local**: `sentence-transformers>=3.0,<4.0` (included in project dependencies)
- **api**: `openai` (included in project dependencies)
- **ONNX** (optional): `pip install optimum[onnxruntime]`

## Why sentence embedding over SequenceMatcher

`difflib.SequenceMatcher` measures character overlap. `"what is your"` → `"what is your name"` scores 0.87 (too similar), blocking re-prepare even though the meaning is completely different. Sentence embedding (all-MiniLM-L6-v2) scores 0.66 for the same pair, correctly identifying it as semantically distinct.
