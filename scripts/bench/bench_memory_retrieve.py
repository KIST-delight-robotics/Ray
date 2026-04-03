"""Retrieve latency benchmark — controlled variable test.

Usage:
    uv run python scripts/bench/bench_memory_retrieve.py <n_episodes> <gap_sec> <query_mode>

    n_episodes: 500 / 2000 / 5000 / 10000
    gap_sec:    0 / 1 / 5
    query_mode: short / long
"""

import sys
import time
import random
import tempfile
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from voice_pipeline.embedding.embedder import SentenceTransformerEmbedder
from voice_pipeline.memory.storage import SQLiteMemoryStorage
from voice_pipeline.memory.vector_index import NumpyVectorIndex
from voice_pipeline.memory.retriever import MemoryRetriever
from voice_pipeline.memory.types import Episode
from voice_pipeline.core.config import MemoryConfig

# --- Args ---
n_episodes = int(sys.argv[1])
gap_sec = float(sys.argv[2])
query_mode = sys.argv[3]  # "short" or "long"

# --- Queries ---
SHORT_QUERIES = [
    "I love watching sci-fi movies",
    "What should I cook for dinner",
    "Tell me about jazz music",
    "I want to travel to Tokyo",
    "Programming and coding projects",
    "Hiking in the mountains",
]

# Simulates _build_retriever_query with 3 turns of history prepended
LONG_QUERIES = [
    "I watched Interstellar last night and it was amazing. That movie always makes me emotional. The space scenes are breathtaking. I love watching sci-fi movies",
    "I tried making pasta yesterday but it didn't turn out well. Maybe I should take a cooking class. Italian food is my favorite though. What should I cook for dinner",
    "Bill Evans is incredible, especially the Waltz for Debby album. I listen to it every morning while coding. Jazz piano is so relaxing. Tell me about jazz music",
    "My friend just came back from Japan and showed me photos. The temples in Kyoto looked stunning. I've always wanted to try authentic ramen there. I want to travel to Tokyo",
    "I've been learning Rust lately and it's challenging but fun. The borrow checker is strict but makes sense. I also use Python for quick scripts. Programming and coding projects",
    "Last weekend I went to Bukhansan and the trail was beautiful. The autumn leaves were at their peak. I try to go every other week. Hiking in the mountains",
]

queries = LONG_QUERIES if query_mode == "long" else SHORT_QUERIES

# --- Setup ---
embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2", expected_dimension=384)
embedder.embed("warmup")

tmpdir = tempfile.mkdtemp()
cfg = MemoryConfig(db_path=os.path.join(tmpdir, "bench.db"), embedding_dimension=384)
storage = SQLiteMemoryStorage(cfg)
index = NumpyVectorIndex()

rng = random.Random(42)
templates = [
    "The user talked about {0} and found it {1}.",
    "The user mentioned {0} with great {1}.",
    "The user said they enjoy {0} because of the {1}.",
]
words_a = ["movies", "cooking", "jazz", "hiking", "chess", "yoga", "Python", "Tokyo", "Dune", "camera"]
words_b = ["excitement", "passion", "interest", "joy", "curiosity", "beauty", "depth"]

texts = [
    rng.choice(templates).format(rng.choice(words_a), rng.choice(words_b)) + f" ({i})"
    for i in range(n_episodes)
]
embs = embedder.embed_batch(texts)
for i, text in enumerate(texts):
    ep = Episode(
        id=None, text=text, timestamp="2026-03-15 14:00:00",
        session_id=f"s-{i // 10}", importance=1.0, last_cited_at="2026-03-15 14:00:00",
    )
    eid = storage.add_episode(ep)
    storage.update_episode_embedding(eid, embs[i])
    index.add(eid, embs[i])

retriever = MemoryRetriever(storage, index, embedder, cfg)

# Warmup retrieve (so retained buffer, FTS cache etc. are primed)
retriever.retrieve("warmup query", set())
time.sleep(gap_sec if gap_sec > 0 else 1.0)  # let cache settle

# --- Measure ---
times = []
for i, q in enumerate(queries):
    if gap_sec > 0:
        time.sleep(gap_sec)
    t0 = time.perf_counter()
    result = retriever.retrieve(q, set())
    ms = (time.perf_counter() - t0) * 1000
    times.append(ms)

storage.close()

# --- Output (tab-separated for easy parsing) ---
for i, ms in enumerate(times):
    print(f"{n_episodes}\t{gap_sec}\t{query_mode}\t{i+1}\t{ms:.1f}")
