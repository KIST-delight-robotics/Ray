"""TurnGPT long-duration stress test.

Tests KV-cache stability, memory growth, latency drift, and eviction
behavior over sustained multi-conversation sessions.

Simulates realistic ASR streaming: words arrive incrementally within a turn,
turns accumulate with <ts> separators, and the dialog eventually hits the
256-token context window triggering eviction. Conversations reset periodically.

Usage:
    uv run python scripts/bench/stress_test_turngpt.py --duration 300
    uv run python scripts/bench/stress_test_turngpt.py --duration 600 --reset-interval 120
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from voice_pipeline.core.config import TurnGPTConfig
from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper

# Realistic multi-turn conversations (varying length and style)
CONVERSATIONS = [
    [
        "hello how are you doing today",
        "i'm doing great thanks for asking how about you",
        "pretty good just been busy with work lately",
        "oh yeah what kind of work do you do",
        "i work in software engineering mostly backend stuff",
        "that sounds interesting do you enjoy it",
        "yeah i really do it's challenging but rewarding",
        "i can imagine what languages do you use",
        "mostly python and go sometimes rust for performance critical stuff",
        "nice i've been wanting to learn rust actually",
    ],
    [
        "hey did you watch the game last night",
        "no i missed it who won",
        "the home team pulled it off in overtime it was incredible",
        "oh man i wish i had seen that what was the final score",
        "it was three to two they scored with just seconds left",
        "that must have been so exciting to watch live",
        "it really was the whole stadium went crazy",
    ],
    [
        "what are you planning for the weekend",
        "i was thinking about going hiking if the weather is nice",
        "oh that sounds fun where would you go",
        "there's a trail about an hour north of here with amazing views",
        "i've been looking for good hiking spots can i come along",
        "of course the more the merrier we usually leave around eight",
        "perfect i'll bring some snacks and water",
        "great let's meet at the parking lot by the trailhead",
    ],
    [
        "i just got back from vacation",
        "oh nice where did you go",
        "we went to japan for two weeks it was amazing",
        "wow two weeks that's a proper trip what was your favorite part",
        "probably the food honestly everything we ate was incredible",
        "i've always wanted to try authentic ramen there",
        "the ramen was unreal we had it almost every day and never got tired of it",
        "you're making me hungry now did you visit tokyo",
        "yeah tokyo osaka kyoto and a few smaller towns",
        "sounds like the perfect itinerary i need to plan a trip",
        "you definitely should i can share our whole route with you",
        "that would be amazing thanks so much",
    ],
    [
        "have you tried that new restaurant downtown",
        "not yet but i heard good things about it",
        "the pasta there is honestly some of the best i've ever had",
        "really i'm always looking for good pasta recommendations",
        "you should go on a weeknight though weekends are packed",
        "good tip i'll try to go this thursday",
    ],
]


def build_incremental_inputs(
    conversations: list[list[str]], *, chain: bool = False,
) -> list[str]:
    """Build a sequence of incremental dialog strings simulating ASR streaming.

    For each utterance, words are added one by one (simulating interim ASR).
    After an utterance completes, a <ts> is added and the next turn begins.
    This produces a realistic stream of growing dialog texts.

    Args:
        chain: If True, dialog history accumulates across all conversations
            (no reset between them), so token count grows past single-
            conversation limits and exercises the eviction window.
    """
    inputs = []
    dialog_parts: list[str] = []

    for conv in conversations:
        if not chain:
            dialog_parts = []
        for utt_idx, utterance in enumerate(conv):
            words = utterance.split()
            for w_idx in range(1, len(words) + 1):
                partial_utt = " ".join(words[:w_idx])
                if dialog_parts:
                    full_text = " <ts> ".join(dialog_parts) + " <ts> " + partial_utt
                else:
                    full_text = partial_utt
                inputs.append(full_text)

            # Utterance complete — add to history
            dialog_parts.append(utterance)

    return inputs


def measure_memory_mb() -> float:
    """Current process RSS in MB."""
    import psutil
    return psutil.Process().memory_info().rss / 1024 / 1024


def main():
    parser = argparse.ArgumentParser(description="TurnGPT long-duration stress test")
    parser.add_argument("--duration", type=float, default=60, help="Test duration (seconds)")
    parser.add_argument("--rate", type=float, default=3.0, help="Calls per second (~ASR rate)")
    parser.add_argument("--reset-interval", type=float, default=60,
                        help="Seconds between conversation resets (0=no reset)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="max_context_tokens (model trained on 256)")
    parser.add_argument("--keep-turns", type=int, default=2,
                        help="Completed turns to keep after eviction")
    parser.add_argument("--onnx-threads", type=int, default=2, help="ONNX intra_op threads")
    parser.add_argument("--int8", action="store_true", help="Use INT8 quantized model")
    args = parser.parse_args()

    model_path = ("models/turngpt/turngpt_v2_kvcache_int8.onnx" if args.int8
                  else "models/turngpt/turngpt_v2_kvcache.onnx")

    print("=" * 70)
    print("  TurnGPT Stress Test")
    print("=" * 70)
    print(f"  Duration         : {args.duration}s")
    print(f"  Rate             : {args.rate} calls/sec")
    print(f"  Reset interval   : {args.reset_interval}s")
    print(f"  Max context      : {args.max_tokens} tokens")
    print(f"  Keep turns       : {args.keep_turns}")
    print(f"  ONNX threads     : {args.onnx_threads}")
    print(f"  Model            : {model_path}")
    print(f"  CPU cores        : {os.cpu_count()}")

    # Load model
    print("\n  Loading TurnGPT (ONNX KV-cache)...")
    config = TurnGPTConfig(
        onnx_model_path=model_path,
        tokenizer_path="models/turngpt/tokenizer",
        onnx_threads=args.onnx_threads,
        max_context_tokens=args.max_tokens,
        keep_turns=args.keep_turns,
    )
    turngpt = TurnGPTWrapper(config)

    # Warmup
    turngpt.predict("hello how are you")
    turngpt.reset()

    # Prepare inputs — chain=True so dialog accumulates across conversations,
    # pushing token count past max_context_tokens to exercise eviction.
    all_inputs = build_incremental_inputs(CONVERSATIONS, chain=True)
    print(f"  Input pool       : {len(all_inputs)} incremental steps")

    # Tokenize a few to show token counts
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("models/turngpt/tokenizer")
    sample_lengths = [len(tok.encode(inp)) for inp in all_inputs]
    print(f"  Token range      : {min(sample_lengths)}–{max(sample_lengths)} tokens")

    interval = 1.0 / args.rate
    mem_start = measure_memory_mb()
    print(f"  RSS at start     : {mem_start:.1f} MB")

    # Tracking
    latencies: list[float] = []
    token_counts: list[int] = []
    timestamps: list[float] = []
    eviction_count = 0
    reset_count = 0
    error_count = 0

    # Per-minute buckets for drift analysis
    minute_buckets: dict[int, list[float]] = {}
    mem_samples: list[tuple[float, float]] = []  # (elapsed_sec, rss_mb)

    tracemalloc.start()

    print(f"\n  Running {args.duration}s stress test...")
    print(f"  {'─' * 60}")

    t_start = time.perf_counter()
    last_reset = t_start
    input_idx = 0
    prev_token_count = 0

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= args.duration:
            break

        t_frame = time.perf_counter()

        # Reset if interval reached
        if args.reset_interval > 0 and (t_frame - last_reset) >= args.reset_interval:
            turngpt.reset()
            last_reset = t_frame
            reset_count += 1
            input_idx = 0  # restart conversation
            prev_token_count = 0

        # Get input
        text = all_inputs[input_idx % len(all_inputs)]
        n_tokens = len(tok.encode(text))

        # Detect eviction (token count drops despite growing input)
        if n_tokens > args.max_tokens * 0.8 and prev_token_count > 0:
            if n_tokens < prev_token_count:
                eviction_count += 1
        prev_token_count = n_tokens

        # Predict
        t0 = time.perf_counter()
        try:
            trp = turngpt.predict(text)
        except Exception as e:
            error_count += 1
            print(f"  ERROR at {elapsed:.1f}s: {e}")
            turngpt.reset()
            input_idx = 0
            prev_token_count = 0
            continue
        lat = (time.perf_counter() - t0) * 1000

        latencies.append(lat)
        token_counts.append(n_tokens)
        timestamps.append(elapsed)

        # Bucket by minute
        minute = int(elapsed // 60)
        minute_buckets.setdefault(minute, []).append(lat)

        # Memory sample every 10 seconds
        if len(mem_samples) == 0 or elapsed - mem_samples[-1][0] >= 10:
            mem_samples.append((elapsed, measure_memory_mb()))

        input_idx += 1

        # Progress
        if len(latencies) % 100 == 0:
            m = int(elapsed // 60)
            s = int(elapsed % 60)
            print(f"    [{m:02d}:{s:02d}] calls={len(latencies)}, "
                  f"last_lat={lat:.1f}ms, tokens={n_tokens}, "
                  f"rss={mem_samples[-1][1]:.1f}MB")

        # Pace
        frame_elapsed = time.perf_counter() - t_frame
        sleep_time = interval - frame_elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    total_elapsed = time.perf_counter() - t_start
    mem_end = measure_memory_mb()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Results
    lats = np.array(latencies)
    toks = np.array(token_counts)

    print(f"\n{'=' * 70}")
    print(f"  Results ({total_elapsed:.1f}s elapsed)")
    print(f"{'=' * 70}")

    print(f"\n  Calls & Errors:")
    print(f"    Total calls    : {len(lats)}")
    print(f"    Resets         : {reset_count}")
    print(f"    Errors         : {error_count}")
    print(f"    Effective rate : {len(lats) / total_elapsed:.1f} calls/sec")

    print(f"\n  Latency (ms):")
    print(f"    Mean           : {lats.mean():.1f}")
    print(f"    Median         : {np.median(lats):.1f}")
    print(f"    P95            : {np.percentile(lats, 95):.1f}")
    print(f"    P99            : {np.percentile(lats, 99):.1f}")
    print(f"    Max            : {lats.max():.1f}")
    print(f"    Std            : {lats.std():.1f}")

    print(f"\n  Token counts:")
    print(f"    Mean           : {toks.mean():.0f}")
    print(f"    Max            : {toks.max()}")
    print(f"    >200 tokens    : {(toks > 200).sum()}/{len(toks)}")

    print(f"\n  Memory:")
    print(f"    RSS start      : {mem_start:.1f} MB")
    print(f"    RSS end        : {mem_end:.1f} MB")
    print(f"    RSS delta      : {mem_end - mem_start:+.1f} MB")
    print(f"    tracemalloc peak: {peak / 1024 / 1024:.1f} MB")

    # Latency by token bucket
    print(f"\n  Latency by token count:")
    for lo, hi in [(1, 20), (20, 50), (50, 100), (100, 150), (150, 200), (200, 260)]:
        mask = (toks >= lo) & (toks < hi)
        if mask.sum() > 0:
            b = lats[mask]
            print(f"    [{lo:3d}–{hi:3d}) n={mask.sum():4d}  "
                  f"mean={b.mean():.1f}  p95={np.percentile(b, 95):.1f}  "
                  f"max={b.max():.1f}")

    # Drift analysis per minute
    print(f"\n  Latency drift (per-minute):")
    print(f"    {'Min':>4s}  {'N':>5s}  {'Mean':>7s}  {'P95':>7s}  {'Max':>7s}")
    for m in sorted(minute_buckets.keys()):
        b = np.array(minute_buckets[m])
        print(f"    {m:4d}  {len(b):5d}  {b.mean():7.1f}  "
              f"{np.percentile(b, 95):7.1f}  {b.max():7.1f}")

    # Spike analysis
    times = np.array(timestamps)
    print(f"\n  Spike analysis:")
    for threshold in [100, 150, 200]:
        spike_mask = lats > threshold
        n_spikes = spike_mask.sum()
        if n_spikes == 0:
            print(f"    >{threshold}ms : 0/{len(lats)} (0.0%)")
            continue
        pct = 100 * n_spikes / len(lats)
        spike_lats = lats[spike_mask]
        spike_toks = toks[spike_mask]
        spike_times = times[spike_mask]
        # Check if spikes cluster or are spread out
        if n_spikes > 1:
            gaps = np.diff(spike_times)
            gap_info = f"  gap mean={gaps.mean():.1f}s min={gaps.min():.1f}s"
        else:
            gap_info = ""
        print(f"    >{threshold}ms : {n_spikes}/{len(lats)} ({pct:.1f}%)"
              f"  lat={spike_lats.mean():.0f}~{spike_lats.max():.0f}ms"
              f"  tok={spike_toks.mean():.0f}~{spike_toks.max()}{gap_info}")

    # Individual spike details (>P99)
    p99 = np.percentile(lats, 99)
    outlier_mask = lats > p99
    if outlier_mask.sum() > 0:
        print(f"\n  Top spikes (>P99={p99:.0f}ms):")
        idxs = np.where(outlier_mask)[0]
        for i in idxs[:10]:
            print(f"    {times[i]:6.1f}s  {lats[i]:6.1f}ms  tokens={toks[i]}")

    # Memory over time
    print(f"\n  Memory over time (RSS MB):")
    for t, mem in mem_samples:
        print(f"    {t:6.0f}s : {mem:.1f} MB")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
