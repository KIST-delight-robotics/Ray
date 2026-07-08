"""Benchmark embedding backends: torch, onnx fp32, onnx qint8 arm64.

Measures per-call latency and cross-backend cosine similarity differences.

Usage:
    uv run python -m scripts.bench.bench_embed [--runs N]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from voice_pipeline.embedding.embedder import SentenceTransformerEmbedder

MODEL = "all-MiniLM-L6-v2"
OUT_DIR = Path(__file__).parent

BACKENDS = {
    "torch": {"use_onnx": False, "model_kwargs": {}},
    "onnx_fp32": {"use_onnx": True, "model_kwargs": {}},
    "onnx_qint8_arm64": {
        "use_onnx": True,
        "model_kwargs": {"file_name": "onnx/model_qint8_arm64.onnx"},
    },
}

LATENCY_TEXTS = [
    "오늘 날씨가 좋아서 기분이 좋아요",
    "내일 회의 준비를 해야 해요",
    "주말에 영화 보러 갈까요",
    "라면이 먹고 싶어요",
    "이번 프로젝트가 거의 끝나가네요",
]

SIMILARITY_PAIRS = [
    ("오늘 날씨 어때?", "오늘 날씨 어떤가요?"),
    ("저녁에 뭐 먹을까", "저녁에 뭐 먹을까요 우리"),
    ("내일 회의 몇 시야", "주말에 뭐 할 거야"),
    ("좋아 알겠어", "좋아 알겠어 그럼"),
    ("이번 주말에 영화 보러 갈까", "이번 주말에"),
    ("라면 먹고 싶다", "라면이 먹고 싶어요"),
    ("오늘 기분이 좋아", "오늘 기분이 안 좋아"),
    ("내일 비 온대", "내일 날씨가 어떨까"),
]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def bench_latency(embedder: SentenceTransformerEmbedder, runs: int) -> list[float]:
    latencies = []
    for _ in range(runs):
        for text in LATENCY_TEXTS:
            t0 = time.perf_counter()
            embedder.embed(text)
            latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def bench_similarity(
    embedders: dict[str, SentenceTransformerEmbedder],
) -> list[dict]:
    rows = []
    for text_a, text_b in SIMILARITY_PAIRS:
        row: dict = {"text_a": text_a, "text_b": text_b}
        vecs_a = {}
        vecs_b = {}
        for name, emb in embedders.items():
            vecs_a[name] = emb.embed(text_a)
            vecs_b[name] = emb.embed(text_b)
            row[f"sim_{name}"] = round(cosine_sim(vecs_a[name], vecs_b[name]), 6)
        names = list(embedders.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n1, n2 = names[i], names[j]
                row[f"vec_sim_a_{n1}_vs_{n2}"] = round(cosine_sim(vecs_a[n1], vecs_a[n2]), 6)
                row[f"vec_sim_b_{n1}_vs_{n2}"] = round(cosine_sim(vecs_b[n1], vecs_b[n2]), 6)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=6)
    args = parser.parse_args()

    embedders: dict[str, SentenceTransformerEmbedder] = {}
    for name, kwargs in BACKENDS.items():
        print(f"Loading {name}...")
        embedders[name] = SentenceTransformerEmbedder(MODEL, **kwargs)

    # -- warmup --
    for emb in embedders.values():
        emb.embed("warmup")

    # -- latency --
    print(f"\nBenchmarking latency ({args.runs} runs x {len(LATENCY_TEXTS)} texts)...")
    latency_path = OUT_DIR / "embed_latency.csv"
    with open(latency_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["backend", "run", "latency_ms"])
        for name, emb in embedders.items():
            latencies = bench_latency(emb, args.runs)
            for i, lat in enumerate(latencies):
                writer.writerow([name, i, round(lat, 2)])
            avg = sum(latencies) / len(latencies)
            med = sorted(latencies)[len(latencies) // 2]
            p95 = sorted(latencies)[int(len(latencies) * 0.95)]
            print(f"  {name:25s}  avg={avg:7.2f}ms  med={med:7.2f}ms  p95={p95:7.2f}ms")

    # -- similarity --
    print("\nBenchmarking similarity...")
    sim_rows = bench_similarity(embedders)
    sim_path = OUT_DIR / "embed_similarity.csv"
    with open(sim_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sim_rows[0].keys())
        writer.writeheader()
        writer.writerows(sim_rows)

    print(f"\n{'Pair':<45s}", end="")
    for name in embedders:
        print(f"  sim_{name:>15s}", end="")
    print()
    print("-" * 130)
    for row in sim_rows:
        label = f"{row['text_a'][:18]}.. / {row['text_b'][:18]}.."
        print(f"{label:<45s}", end="")
        for name in embedders:
            print(f"  {row[f'sim_{name}']:>19.6f}", end="")
        print()

    print(f"\n{'Pair':<45s}", end="")
    names = list(embedders.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(f"  vec_a_{names[i][:5]}_vs_{names[j][:5]}", end="")
            print(f"  vec_b_{names[i][:5]}_vs_{names[j][:5]}", end="")
    print()
    print("-" * 180)
    for row in sim_rows:
        label = f"{row['text_a'][:18]}.. / {row['text_b'][:18]}.."
        print(f"{label:<45s}", end="")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ka = f"vec_sim_a_{names[i]}_vs_{names[j]}"
                kb = f"vec_sim_b_{names[i]}_vs_{names[j]}"
                print(f"  {row[ka]:>19.6f}  {row[kb]:>19.6f}", end="")
        print()

    print(f"\nResults saved to {latency_path} and {sim_path}")


if __name__ == "__main__":
    main()
