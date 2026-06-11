"""E2E evaluation runner for the voice pipeline.

Plays pre-generated question WAV files through a physical speaker,
lets the full pipeline process them, and records session mappings.
Actual turn data (ASR text, response, latency) is captured by the
existing storage systems (trace_store, history, memory_storage).

Usage:
    uv run python scripts/eval/run.py \\
        --questions data/eval/questions.json \\
        --device plughw:1,0 \\
        --output-dir data/eval/results
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import queue
import random
import signal
import sys
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Suppress ALSA/JACK noise during PyAudio initialization.
_alsa_error_handler = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
)(lambda *_: None)
try:
    _asound = ctypes.cdll.LoadLibrary("libasound.so.2")
    _asound.snd_lib_error_set_handler(_alsa_error_handler)
except Exception:
    _asound = None

import torch
from question_player import QuestionPlayer
from silero_vad import load_silero_vad

from voice_pipeline.asr.asr import GoogleCloudASR
from voice_pipeline.audio.audio_input import AudioInput
from voice_pipeline.audio.constants import SAMPLE_RATE
from voice_pipeline.bridge.cpp_bridge import CppBridge
from voice_pipeline.core.types import AudioFrame
from voice_pipeline.embedding.embedder import create_embedder
from voice_pipeline.generation.speech_generator import SpeechGenerator
from voice_pipeline.history.conversation_history import ConversationHistory
from voice_pipeline.history.storage_backend import create_storage_backend
from voice_pipeline.led.led_controller import LEDController
from voice_pipeline.llm.llm import OpenAILLM
from voice_pipeline.llm.prompts import DEFAULT_SYSTEM_PROMPT
from voice_pipeline.llm.token_counter import create_token_counter
from voice_pipeline.llm.tools import get_tools_token_cost
from voice_pipeline.memory.retriever import MemoryRetriever
from voice_pipeline.memory.storage import _DEFAULT_DIMENSION, SQLiteMemoryStorage
from voice_pipeline.memory.vector_index import NumpyVectorIndex
from voice_pipeline.session_loop import SessionComponents, SessionLoop
from voice_pipeline.trace.openai_retry_handler import OpenAIRetryHandler
from voice_pipeline.trace.trace_store import SQLiteCallStore, SQLiteTraceStore
from voice_pipeline.trace.tracked_embedder import TrackedEmbedder
from voice_pipeline.trace.tracked_tts import TrackedTTS
from voice_pipeline.tts.tts import OpenAITTS
from voice_pipeline.turn_taking.async_turngpt import AsyncTurnGPT
from voice_pipeline.turn_taking.async_vap import AsyncVAP
from voice_pipeline.turn_taking.maai_vap import MaAIVAPWrapper
from voice_pipeline.turn_taking.turn_detector import TurnDetector
from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper

logger = logging.getLogger("eval")

_AUDIO_QUEUE_SIZE = 300
_STARTUP_DELAY_SEC = 1.5
_TURN_TIMEOUT_SEC = 60.0
_TURN_DETECT_TIMEOUT_SEC = 10.0
_WATCHDOG_POLL_SEC = 0.1  # 감지 워치독 폴링 주기
_BEEP_SETTLE_SEC = 0.2  # 비프음이 마이크 큐에 다 들어온 뒤 drain하기 위한 대기
_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drain_audio_queue(audio_queue: queue.Queue[AudioFrame]) -> None:
    while True:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


def _begin_session_audio(player: QuestionPlayer, audio_queue: queue.Queue[AudioFrame]) -> None:
    """Sound the session-start beep, then clear the audio queue.

    ``audio_input`` runs continuously across the whole eval, so the beep —
    played through the same external speaker as the questions — is picked up
    by the mic. Draining after it (once the beep has settled into the queue)
    keeps it out of the ASR stream and the session recording, and also clears
    any frames left over from the previous session.
    """
    if player.beep():
        time.sleep(_BEEP_SETTLE_SEC)
    _drain_audio_queue(audio_queue)


def _make_beep_wav(
    path: str,
    *,
    freq_hz: float = 1000.0,
    duration_sec: float = 0.15,
    sample_rate: int = 16000,
    volume: float = 0.5,
) -> None:
    """Write a short sine-wave beep (mono 16-bit) used to mark session starts.

    Played through the question speaker just before each session's mic
    capture begins, so a human watching the run can hear when a new
    question/scenario starts. Endpoints are faded to avoid click artifacts.
    """
    import math
    import struct
    import wave

    n = int(sample_rate * duration_sec)
    fade = max(1, int(sample_rate * 0.01))
    frames = bytearray()
    for i in range(n):
        amp = volume
        if i < fade:
            amp *= i / fade
        elif i >= n - fade:
            amp *= (n - i) / fade
        sample = int(amp * 32767 * math.sin(2 * math.pi * freq_hz * i / sample_rate))
        frames += struct.pack("<h", sample)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))


def _iter_questions(suite: dict):
    """Yield all question dicts from a suite (handles both flat and scenario formats)."""
    if suite.get("multi_turn"):
        for scenario in suite.get("scenarios", []):
            yield from scenario["questions"]
    else:
        yield from suite.get("questions", [])


def _load_manifest(questions_path: str, wav_dir: str) -> dict[str, dict[str, str]]:
    """Load WAV manifest. Returns ``{id: {"path": ..., "voice": ...}}``.

    Supports both the new format (``{id: {"path", "voice"}}``) and the legacy
    format (``{id: path_str}``).  Falls back to ``<wav_dir>/<id>.wav`` when no
    manifest file exists.
    """
    manifest_path = Path(wav_dir) / "manifest.json"
    if manifest_path.exists():
        raw = json.loads(manifest_path.read_text())
        first = next(iter(raw.values()), None) if raw else None
        if isinstance(first, str):
            return {qid: {"path": p, "voice": ""} for qid, p in raw.items()}
        return raw
    data = json.loads(Path(questions_path).read_text())
    manifest: dict[str, dict[str, str]] = {}
    wav_dir_path = Path(wav_dir)
    for suite in data["suites"]:
        for q in _iter_questions(suite):
            legacy = wav_dir_path / f"{q['id']}.wav"
            if legacy.exists():
                manifest[q["id"]] = {"path": str(legacy), "voice": ""}
                continue
            matches = sorted(wav_dir_path.glob(f"{q['id']}_*.wav"))
            if matches:
                voice = matches[0].stem.removeprefix(f"{q['id']}_")
                manifest[q["id"]] = {"path": str(matches[0]), "voice": voice}
    return manifest


def _inject_seeds(seed_data, memory_storage, vector_index, embedder, token_counter):
    """Inject seed sessions into memory storage and run MemoryWriter on each.

    Returns:
        dict[int, str]: Mapping from session index to session_id.
    """
    from voice_pipeline.llm.llm import OpenAILLM as _OpenAILLM
    from voice_pipeline.memory.writer import MemoryWriter

    write_llm = _OpenAILLM(
        model="gpt-4o-mini",
        temperature=0.0,
        reasoning_effort=None,
        max_tokens=4096,
        tools=[],
    )
    writer = MemoryWriter(memory_storage, vector_index, embedder, write_llm, token_counter)

    session_map: dict[int, str] = {}
    for idx, session in enumerate(seed_data["sessions"]):
        session_id = f"seed_{uuid.uuid4()}"
        timestamp = f"{session['date']} 10:00:00"
        for utt in session["utterances"]:
            memory_storage.add_utterance(
                session_id,
                utt["role"],
                utt["text"],
                timestamp,
                token_counter(utt["text"]),
            )
        writer.process_session(session_id, timestamp)
        session_map[idx] = session_id
        logger.info("Seed session %d → %s (%s)", idx, session_id, session["date"])

    return session_map


def _resolve_target_episodes(target_sessions, seed_session_map, seed_episode_map):
    """Map target session indices to episode IDs."""
    ids = []
    for idx in target_sessions:
        sid = seed_session_map.get(idx)
        if sid:
            ids.extend(seed_episode_map.get(sid, []))
    return ids


class _PauseController:
    """Listens for Enter key on stdin to toggle pause/resume between sessions."""

    def __init__(self, shutdown_event: threading.Event) -> None:
        self._shutdown = shutdown_event
        self._pause_requested = threading.Event()
        self._resume = threading.Event()
        self._paused = threading.Event()

    def start(self) -> None:
        if not sys.stdin.isatty():
            return
        thread = threading.Thread(target=self._listen, daemon=True)
        thread.start()
        logger.info("Press Enter to pause/resume evaluation between sessions")

    def _listen(self) -> None:
        while not self._shutdown.is_set():
            try:
                line = sys.stdin.readline()
                if not line:
                    break
            except (OSError, ValueError):
                break
            if self._paused.is_set():
                self._resume.set()
            else:
                self._pause_requested.set()
                logger.info("Pause requested — will pause after current session completes")

    def wait_if_paused(self) -> bool:
        """Call at session boundaries. Returns True if eval should stop."""
        if self._shutdown.is_set():
            return True
        if not self._pause_requested.is_set():
            return False
        self._pause_requested.clear()
        self._paused.set()
        logger.info("Paused. Press Enter to resume...")
        while not self._resume.is_set():
            if self._shutdown.is_set():
                self._paused.clear()
                return True
            self._resume.wait(timeout=0.5)
        self._resume.clear()
        self._paused.clear()
        logger.info("Resumed")
        return False


# ---------------------------------------------------------------------------
# Text-mode execution
# ---------------------------------------------------------------------------


def _run_text_single_turn(suite, question, session_map, create_text_session, seed_episode_map, seed_session_map):
    """Execute one single-turn eval question in text mode."""
    is_memory = suite.get("category") == "memory"
    session = create_text_session(
        memory_enabled=suite.get("memory", False),
        load_session_context=not is_memory,
    )
    try:
        session.send(question["text"])
    except Exception:
        logger.error("Text session error for %s", question["id"], exc_info=True)
        session_map.append(
            {
                "question_id": question["id"],
                "session_id": session.session_id,
                "suite_name": suite["name"],
                "input_text": question["text"],
                "asr_text": question["text"],
                "success": False,
                "error": "text_session_error",
                "text_mode": True,
                "retrieved_episodes": [],
                "target_sessions": question.get("target_sessions", []),
                "target_episode_ids": _resolve_target_episodes(
                    question.get("target_sessions", []), seed_session_map, seed_episode_map
                ),
            }
        )
        session.close()
        return

    # Extract retrieved episodes from memory results
    retrieved_episodes = []
    if session.memory_results and session.memory_results[-1] is not None:
        mr = session.memory_results[-1]
        for ep, score in zip(mr.episodes, mr.scores, strict=False):
            retrieved_episodes.append(
                {
                    "episode_id": ep.id,
                    "text": ep.text,
                    "score": round(score, 4),
                    "timestamp": ep.timestamp,
                    "session_id": ep.session_id,
                }
            )

    target_episode_ids = _resolve_target_episodes(
        question.get("target_sessions", []), seed_session_map, seed_episode_map
    )

    trace = session.traces[-1] if session.traces else None
    latency = {}
    if trace:
        latency = {
            "memory_ms": round(trace.memory_ms, 1),
            "context_ms": round(trace.context_ms, 1),
            "llm_ms": round(trace.llm_ms, 1),
            "llm_ttft_ms": round(trace.llm_ttft_ms, 1),
            "total_ms": round(trace.total_ms, 1),
        }

    session_map.append(
        {
            "question_id": question["id"],
            "session_id": session.session_id,
            "suite_name": suite["name"],
            "input_text": question["text"],
            "asr_text": question["text"],
            "success": True,
            "error": None,
            "text_mode": True,
            "latency": latency,
            "retrieved_episodes": retrieved_episodes,
            "target_sessions": question.get("target_sessions", []),
            "target_episode_ids": target_episode_ids,
        }
    )
    session.close()

    logger.info("[OK] %s: %s", question["id"], question["text"][:60])


def _run_text_multi_turn(suite, scenario, session_map, create_text_session, seed_episode_map, seed_session_map):
    """Execute a multi-turn scenario in text mode (one session for all turns)."""
    is_memory = suite.get("category") == "memory"
    session = create_text_session(
        memory_enabled=suite.get("memory", False),
        load_session_context=not is_memory,
    )
    questions = scenario["questions"]

    try:
        for i, question in enumerate(questions):
            try:
                session.send(question["text"])
            except Exception:
                logger.error("Text session error at turn %d for %s", i, question["id"], exc_info=True)
                session_map.append(
                    {
                        "question_id": question["id"],
                        "scenario_id": scenario["id"],
                        "session_id": session.session_id,
                        "suite_name": suite["name"],
                        "input_text": question["text"],
                        "asr_text": question["text"],
                        "success": False,
                        "error": "text_session_error",
                        "text_mode": True,
                        "retrieved_episodes": [],
                        "target_sessions": question.get("target_sessions", []),
                        "target_episode_ids": _resolve_target_episodes(
                            question.get("target_sessions", []), seed_session_map, seed_episode_map
                        ),
                    }
                )
                continue

            # Extract retrieved episodes for this turn
            retrieved_episodes = []
            turn_idx = i  # memory_results index matches send() call order
            if len(session.memory_results) > turn_idx and session.memory_results[turn_idx] is not None:
                mr = session.memory_results[turn_idx]
                for ep, score in zip(mr.episodes, mr.scores, strict=False):
                    retrieved_episodes.append(
                        {
                            "episode_id": ep.id,
                            "text": ep.text,
                            "score": round(score, 4),
                            "timestamp": ep.timestamp,
                            "session_id": ep.session_id,
                        }
                    )

            target_episode_ids = _resolve_target_episodes(
                question.get("target_sessions", []), seed_session_map, seed_episode_map
            )

            trace = session.traces[turn_idx] if len(session.traces) > turn_idx else None
            latency = {}
            if trace:
                latency = {
                    "memory_ms": round(trace.memory_ms, 1),
                    "context_ms": round(trace.context_ms, 1),
                    "llm_ms": round(trace.llm_ms, 1),
                    "llm_ttft_ms": round(trace.llm_ttft_ms, 1),
                    "total_ms": round(trace.total_ms, 1),
                }

            session_map.append(
                {
                    "question_id": question["id"],
                    "scenario_id": scenario["id"],
                    "session_id": session.session_id,
                    "suite_name": suite["name"],
                    "input_text": question["text"],
                    "asr_text": question["text"],
                    "success": True,
                    "error": None,
                    "text_mode": True,
                    "latency": latency,
                    "retrieved_episodes": retrieved_episodes,
                    "target_sessions": question.get("target_sessions", []),
                    "target_episode_ids": target_episode_ids,
                }
            )
    finally:
        session.close()

    logger.info(
        "Scenario %s (%s): %d turns completed (text mode)",
        scenario["id"],
        suite["name"],
        len(questions),
    )


# ---------------------------------------------------------------------------
# Single-turn execution
# ---------------------------------------------------------------------------


def _run_single_turn(
    suite: dict,
    question: dict,
    wav_path: str,
    player: QuestionPlayer,
    session_map: list[dict],
    create_session: Callable[..., SessionComponents],
    audio_queue: queue.Queue[AudioFrame],
    seed_session_map: dict[int, str] | None = None,
    seed_episode_map: dict[str, list[int]] | None = None,
    record_dir: str | None = None,
    voice: str = "",
) -> None:
    """Execute one single-turn eval: play question, capture response."""
    _begin_session_audio(player, audio_queue)

    turn_event = threading.Event()
    gen_failed_event = threading.Event()
    watchdog_stop = threading.Event()
    play_end_time = [0.0]
    turn_shift_time = [0.0]
    final_asr_text = [""]
    shift_count = [0]
    cancel_count = [0]

    def on_turn_done(ts_time: float, asr_text: str) -> None:
        turn_shift_time[0] = ts_time
        final_asr_text[0] = asr_text
        turn_event.set()
        components.session_loop.request_stop()

    def on_turn_shift(ts_time: float, asr_text: str) -> None:
        shift_count[0] += 1

    def on_cancel() -> None:
        cancel_count[0] += 1

    def on_gen_failed() -> None:
        gen_failed_event.set()
        components.session_loop.request_stop()

    rec_path = str(Path(record_dir) / f"{question['id']}.wav") if record_dir else None
    skip = suite.get("category") == "asr"
    components = create_session(
        on_turn_complete=on_turn_done,
        on_turn_shift=on_turn_shift,
        on_generation_failed=on_gen_failed,
        on_cancel=on_cancel,
        memory_enabled=suite.get("memory", False),
        skip_generation=skip,
        record_path=rec_path,
    )
    components.history.new_session(components.session_id)

    def play_with_delay() -> None:
        time.sleep(_STARTUP_DELAY_SEC)
        try:
            player.play(wav_path)
            play_end_time[0] = time.monotonic()
        except Exception:
            logger.error("Failed to play %s", wav_path, exc_info=True)
            return
        # 감지 타임아웃은 질문 재생 종료 시점에 고정 — 잠정 shift가 cancel로
        # 철회되면 같은 기준선으로 재대기. 잠정 shift가 서 있는 동안(생성 중)은
        # 타임아웃을 평가하지 않음 (생성 지연을 감지 실패로 오분류 방지).
        deadline = play_end_time[0] + _TURN_DETECT_TIMEOUT_SEC
        while not turn_event.is_set() and not watchdog_stop.is_set():
            standing_shift = shift_count[0] > cancel_count[0]
            if not standing_shift and time.monotonic() >= deadline:
                logger.warning("Turn detection timeout for %s", question["id"])
                components.session_loop.request_stop()
                return
            time.sleep(_WATCHDOG_POLL_SEC)

    player_thread = threading.Thread(target=play_with_delay, daemon=True)
    player_thread.start()

    try:
        components.session_loop.run()
    except Exception:
        logger.error("SessionLoop error", exc_info=True)
    finally:
        components.history.save()
        watchdog_stop.set()

    player_thread.join(timeout=5.0)

    success = turn_event.is_set()
    gen_failed = gen_failed_event.is_set()
    early_turn_shift = success and play_end_time[0] == 0.0
    ts_reason = components.session_loop.turn_shift_reason
    # expect_wait 스위트(미완성 발화)는 최대 timeout까지 대기한 전환만 정답 —
    # 그보다 빠른 전환은 미완성 발화를 완결로 오판한 것(premature).
    expect_wait = suite.get("expect_wait", False)
    completed_normally = success and not early_turn_shift
    if expect_wait:
        late_turn_shift = False
        premature_turn_shift = completed_normally and ts_reason != "turngpt_3.0"
    else:
        late_turn_shift = completed_normally and ts_reason == "turngpt_3.0"
        premature_turn_shift = False
    vap_delay = None
    if not early_turn_shift and play_end_time[0] > 0 and turn_shift_time[0] >= play_end_time[0]:
        vap_delay = round((turn_shift_time[0] - play_end_time[0]) * 1000, 1)

    error = None
    if gen_failed:
        error = "generation_failed"
    elif not success:
        error = "no_turn_shift" if final_asr_text[0] else "no_recognition"
    elif early_turn_shift:
        error = "early_turn_shift"
    elif late_turn_shift:
        error = "late_turn_shift"
    elif premature_turn_shift:
        error = "premature_turn_shift"

    # Extract retrieved episodes from memory results
    retrieved_episodes = []
    mem_results = components.session_loop.memory_results
    if mem_results and mem_results[-1] is not None:
        mr = mem_results[-1]
        for ep, score in zip(mr.episodes, mr.scores, strict=False):
            retrieved_episodes.append(
                {
                    "episode_id": ep.id,
                    "text": ep.text,
                    "score": round(score, 4),
                    "timestamp": ep.timestamp,
                    "session_id": ep.session_id,
                }
            )

    target_episode_ids = _resolve_target_episodes(
        question.get("target_sessions", []), seed_session_map or {}, seed_episode_map or {}
    )

    entry = {
        "question_id": question["id"],
        "session_id": components.session_id,
        "suite_name": suite["name"],
        "input_text": question["text"],
        "asr_text": final_asr_text[0],
        "voice": voice,
        "success": success
        and not early_turn_shift
        and not late_turn_shift
        and not premature_turn_shift
        and not gen_failed,
        "error": error,
        "turn_shift_reason": ts_reason,
        "turn_detection_delay_ms": vap_delay,
    }
    if expect_wait:
        entry["expect_wait"] = True
    if retrieved_episodes:
        entry["retrieved_episodes"] = retrieved_episodes
    if target_episode_ids:
        entry["target_episode_ids"] = target_episode_ids
        entry["target_sessions"] = question.get("target_sessions", [])
    session_map.append(entry)

    status = "OK" if success else "FAIL"
    logger.info("[%s] %s: %s", status, question["id"], question["text"][:60])


# ---------------------------------------------------------------------------
# Interruption execution
# ---------------------------------------------------------------------------


def _run_interruption(
    suite: dict,
    question: dict,
    wav_path: str,
    interrupt_wav_path: str,
    interrupt_delay_sec: float,
    player: QuestionPlayer,
    session_map: list[dict],
    create_session: Callable[..., SessionComponents],
    audio_queue: queue.Queue[AudioFrame],
    interrupt_audio: str = "",
    seed_session_map: dict[int, str] | None = None,
    seed_episode_map: dict[str, list[int]] | None = None,
    record_dir: str | None = None,
    voice: str = "",
) -> None:
    """Execute one interruption test: play question, wait for response, interrupt."""
    _begin_session_audio(player, audio_queue)

    turn_event = threading.Event()
    gen_failed_event = threading.Event()
    playback_started_event = threading.Event()
    watchdog_stop = threading.Event()
    play_end_time = [0.0]
    turn_shift_time = [0.0]
    interrupt_played = [False]
    shift_count = [0]
    cancel_count = [0]

    def on_turn_done(ts_time: float, asr_text: str) -> None:
        turn_shift_time[0] = ts_time
        turn_event.set()
        components.session_loop.request_stop()

    def on_turn_shift(ts_time: float, asr_text: str) -> None:
        shift_count[0] += 1

    def on_cancel() -> None:
        cancel_count[0] += 1

    def on_gen_failed() -> None:
        gen_failed_event.set()
        components.session_loop.request_stop()

    def on_playback_started() -> None:
        playback_started_event.set()

    rec_path = (
        str(Path(record_dir) / f"{question['id']}_{interrupt_audio}_{interrupt_delay_sec:.0f}s.wav")
        if record_dir
        else None
    )
    components = create_session(
        on_turn_complete=on_turn_done,
        on_turn_shift=on_turn_shift,
        on_playback_started=on_playback_started,
        on_generation_failed=on_gen_failed,
        on_cancel=on_cancel,
        memory_enabled=suite.get("memory", False),
        record_path=rec_path,
    )
    components.history.new_session(components.session_id)

    def play_and_interrupt() -> None:
        time.sleep(_STARTUP_DELAY_SEC)
        try:
            player.play(wav_path)
            play_end_time[0] = time.monotonic()
        except Exception:
            logger.error("Failed to play %s", wav_path, exc_info=True)
            return

        # 감지 타임아웃은 질문 재생 종료 시점에 고정 — 잠정 shift가 cancel로
        # 철회되면 같은 기준선으로 재대기 (tt 러너와 동일). 재생 시작 대기 중
        # cancel이 나면 감지 대기로 복귀한다.
        detect_deadline = play_end_time[0] + _TURN_DETECT_TIMEOUT_SEC
        playback_deadline = play_end_time[0] + _TURN_TIMEOUT_SEC
        while not playback_started_event.is_set():
            if turn_event.is_set() or watchdog_stop.is_set():
                return
            standing_shift = shift_count[0] > cancel_count[0]
            now = time.monotonic()
            if not standing_shift and now >= detect_deadline:
                logger.warning("Turn detection timeout for %s", question["id"])
                components.session_loop.request_stop()
                return
            if standing_shift and now >= playback_deadline:
                logger.warning("Playback never started for %s", question["id"])
                components.session_loop.request_stop()
                return
            time.sleep(_WATCHDOG_POLL_SEC)
        time.sleep(interrupt_delay_sec)

        try:
            player.play(interrupt_wav_path)
            interrupt_played[0] = True
        except Exception:
            logger.error("Failed to play interrupt %s", interrupt_wav_path, exc_info=True)

    player_thread = threading.Thread(target=play_and_interrupt, daemon=True)
    player_thread.start()

    try:
        components.session_loop.run()
    except Exception:
        logger.error("SessionLoop error", exc_info=True)
    finally:
        components.history.save()
        watchdog_stop.set()

    player_thread.join(timeout=10.0)

    success = turn_event.is_set()
    gen_failed = gen_failed_event.is_set()
    ts_reason = components.session_loop.turn_shift_reason
    vap_delay = None
    if play_end_time[0] > 0 and turn_shift_time[0] >= play_end_time[0]:
        vap_delay = round((turn_shift_time[0] - play_end_time[0]) * 1000, 1)

    # Extract retrieved episodes
    retrieved_episodes = []
    mem_results = components.session_loop.memory_results
    if mem_results and mem_results[-1] is not None:
        mr = mem_results[-1]
        for ep, score in zip(mr.episodes, mr.scores, strict=False):
            retrieved_episodes.append(
                {
                    "episode_id": ep.id,
                    "text": ep.text,
                    "score": round(score, 4),
                    "timestamp": ep.timestamp,
                    "session_id": ep.session_id,
                }
            )

    target_episode_ids = _resolve_target_episodes(
        question.get("target_sessions", []), seed_session_map or {}, seed_episode_map or {}
    )

    entry = {
        "question_id": question["id"],
        "session_id": components.session_id,
        "suite_name": suite["name"],
        "input_text": question["text"],
        "voice": voice,
        "interrupt_audio": interrupt_audio,
        "interrupt_delay_sec": interrupt_delay_sec,
        "interrupt_played": interrupt_played[0],
        "success": success and not gen_failed,
        "error": "generation_failed" if gen_failed else (None if success else "no_turn_shift"),
        "turn_shift_reason": ts_reason,
        "turn_detection_delay_ms": vap_delay,
    }
    if retrieved_episodes:
        entry["retrieved_episodes"] = retrieved_episodes
    if target_episode_ids:
        entry["target_episode_ids"] = target_episode_ids
        entry["target_sessions"] = question.get("target_sessions", [])
    session_map.append(entry)

    status = "OK" if success else "FAIL"
    logger.info(
        "[%s] %s: delay=%.1fs int=%s played=%s",
        status,
        question["id"],
        interrupt_delay_sec,
        interrupt_audio,
        interrupt_played[0],
    )


# ---------------------------------------------------------------------------
# Multi-turn execution
# ---------------------------------------------------------------------------


def _run_multi_turn_suite(
    suite: dict,
    scenario: dict,
    wav_map: dict[str, dict[str, str]],
    player: QuestionPlayer,
    session_map: list[dict],
    create_session: Callable[..., SessionComponents],
    audio_queue: queue.Queue[AudioFrame],
    seed_session_map: dict[int, str] | None = None,
    seed_episode_map: dict[str, list[int]] | None = None,
    record_dir: str | None = None,
) -> None:
    """Execute a single multi-turn scenario: all questions in one session."""
    _begin_session_audio(player, audio_queue)

    questions = scenario["questions"]
    turn_event = threading.Event()
    gen_failed_event = threading.Event()
    watchdog_stop = threading.Event()
    turn_index = [0]
    shift_count = [0]
    cancel_count = [0]
    turn_shift_times: dict[int, float] = {}
    turn_asr_texts: dict[int, str] = {}
    turn_shift_reasons: dict[int, str | None] = {}

    def on_turn_done(ts_time: float, asr_text: str) -> None:
        turn_shift_times[turn_index[0]] = ts_time
        turn_asr_texts[turn_index[0]] = asr_text
        turn_shift_reasons[turn_index[0]] = components.session_loop.turn_shift_reason
        turn_index[0] += 1
        if turn_index[0] >= len(questions):
            components.session_loop.request_stop()
        turn_event.set()

    def on_turn_shift(ts_time: float, asr_text: str) -> None:
        shift_count[0] += 1

    def on_cancel() -> None:
        cancel_count[0] += 1

    def on_gen_failed() -> None:
        gen_failed_event.set()
        components.session_loop.request_stop()

    rec_path = str(Path(record_dir) / f"{scenario['id']}.wav") if record_dir else None
    skip = suite.get("category") == "asr"
    components = create_session(
        on_turn_complete=on_turn_done,
        on_turn_shift=on_turn_shift,
        on_generation_failed=on_gen_failed,
        on_cancel=on_cancel,
        memory_enabled=suite.get("memory", False),
        skip_generation=skip,
        record_path=rec_path,
    )
    components.history.new_session(components.session_id)

    play_end_times: dict[str, float] = {}

    def play_sequence() -> None:
        time.sleep(_STARTUP_DELAY_SEC)
        for i, q in enumerate(questions):
            wav_entry = wav_map.get(q["id"])
            wav_path = wav_entry["path"] if wav_entry else None
            if wav_path is None:
                logger.error("No WAV for question %s", q["id"])
                break
            if i > 0:
                turn_event.clear()
            base_shift = shift_count[0]
            base_cancel = cancel_count[0]
            try:
                player.play(wav_path)
                play_end_times[q["id"]] = time.monotonic()
            except Exception:
                logger.error("Failed to play %s", wav_path, exc_info=True)
                break
            # 감지 타임아웃은 질문 재생 종료 시점에 고정 — cancel로 잠정 shift가
            # 철회되면 같은 기준선으로 재대기. 잠정 shift가 서 있는 동안은
            # 감지 타임아웃을 평가하지 않고, 턴 완료 상한만 적용.
            detect_deadline = play_end_times[q["id"]] + _TURN_DETECT_TIMEOUT_SEC
            complete_deadline = play_end_times[q["id"]] + _TURN_TIMEOUT_SEC
            failed = None
            while not turn_event.is_set() and not watchdog_stop.is_set():
                now = time.monotonic()
                standing_shift = (shift_count[0] - base_shift) > (cancel_count[0] - base_cancel)
                if not standing_shift and now >= detect_deadline:
                    failed = "detection"
                    break
                if now >= complete_deadline:
                    failed = "completion"
                    break
                time.sleep(_WATCHDOG_POLL_SEC)
            if watchdog_stop.is_set():
                break
            if failed:
                logger.warning("Turn %s timeout at question %s", failed, q["id"])
                components.session_loop.request_stop()
                break

    player_thread = threading.Thread(target=play_sequence, daemon=True)
    player_thread.start()

    try:
        components.session_loop.run()
    except Exception:
        logger.error("SessionLoop error", exc_info=True)
    finally:
        components.history.save()
        watchdog_stop.set()

    player_thread.join(timeout=5.0)

    completed_turns = turn_index[0]
    scenario_voice = wav_map.get(questions[0]["id"], {}).get("voice", "") if questions else ""
    mem_results = components.session_loop.memory_results
    for i, q in enumerate(questions):
        vap_delay = None
        pe = play_end_times.get(q["id"], 0.0)
        ts = turn_shift_times.get(i, 0.0)
        if pe > 0 and ts >= pe:
            vap_delay = round((ts - pe) * 1000, 1)

        retrieved_episodes = []
        if mem_results and i < len(mem_results) and mem_results[i] is not None:
            mr = mem_results[i]
            for ep, score in zip(mr.episodes, mr.scores, strict=False):
                retrieved_episodes.append(
                    {
                        "episode_id": ep.id,
                        "text": ep.text,
                        "score": round(score, 4),
                        "timestamp": ep.timestamp,
                        "session_id": ep.session_id,
                    }
                )

        target_episode_ids = _resolve_target_episodes(
            q.get("target_sessions", []), seed_session_map or {}, seed_episode_map or {}
        )

        ts_reason = turn_shift_reasons.get(i)
        is_completed = i < completed_turns
        expect_wait = suite.get("expect_wait", False)
        if expect_wait:
            is_late = False
            is_premature = is_completed and ts_reason != "turngpt_3.0"
        else:
            is_late = is_completed and ts_reason == "turngpt_3.0"
            is_premature = False
        is_gen_failed = not is_completed and gen_failed_event.is_set() and i == completed_turns

        if is_gen_failed:
            error = "generation_failed"
        elif not is_completed:
            error = "no_turn_shift" if turn_asr_texts.get(i) else "no_recognition"
        elif is_late:
            error = "late_turn_shift"
        elif is_premature:
            error = "premature_turn_shift"
        else:
            error = None

        entry = {
            "question_id": q["id"],
            "scenario_id": scenario["id"],
            "session_id": components.session_id,
            "suite_name": suite["name"],
            "input_text": q["text"],
            "asr_text": turn_asr_texts.get(i, ""),
            "voice": scenario_voice,
            "success": is_completed and not is_late and not is_premature,
            "error": error,
            "turn_shift_reason": ts_reason,
            "turn_detection_delay_ms": vap_delay,
        }
        if expect_wait:
            entry["expect_wait"] = True
        if retrieved_episodes:
            entry["retrieved_episodes"] = retrieved_episodes
        if target_episode_ids:
            entry["target_episode_ids"] = target_episode_ids
            entry["target_sessions"] = q.get("target_sessions", [])
        session_map.append(entry)

    logger.info(
        "Scenario %s (%s): %d/%d turns completed",
        scenario["id"],
        suite["name"],
        completed_turns,
        len(questions),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E evaluation runner")
    parser.add_argument("--questions", required=True, help="Path to questions JSON")
    parser.add_argument("--device", default="default", help="ALSA device for question playback")
    parser.add_argument("--output-dir", default="data/eval/results", help="Output directory")
    parser.add_argument("--wav-dir", default="data/eval/wav", help="Directory with question WAVs")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 1 question per suite")
    parser.add_argument("--category", default=None, help="Only run suites of these categories (comma-separated)")
    parser.add_argument("--text", action="store_true", help="Run quality/memory suites in text mode")
    parser.add_argument("--no-beep", action="store_true", help="Disable the session-start identification beep")
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s %(name)-40s %(levelname)-7s %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    file_handler = logging.FileHandler(output_dir / "eval.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger("voice_pipeline").setLevel(logging.DEBUG)
    logging.getLogger().addHandler(file_handler)

    for entry in os.environ.get("LOG_LEVEL", "").split(","):
        entry = entry.strip()
        if "=" in entry:
            name, level = entry.split("=", 1)
            logging.getLogger(name.strip()).setLevel(level.strip().upper())

    # --- Load questions & seeds ---
    questions_data = json.loads(Path(args.questions).read_text())

    seed_data = None
    seed_file = questions_data.get("seed_file")
    seed_path = None
    if seed_file:
        seed_path = Path(args.questions).parent / seed_file
        if seed_path.exists():
            seed_data = json.loads(seed_path.read_text())

    # --- Classify suites into text vs audio ---
    text_categories: set[str] = set()
    if args.text:
        text_categories = {"quality", "memory"}

    all_suites = [s for s in questions_data["suites"] if not s.get("category", "").startswith("_")]
    if args.category:
        selected = {c.strip() for c in args.category.split(",")}
        available = {s.get("category") for s in all_suites}
        unknown = selected - available
        if unknown:
            logger.error(
                "Unknown categories: %s (available: %s)", ", ".join(sorted(unknown)), ", ".join(sorted(available))
            )
            sys.exit(1)
        all_suites = [s for s in all_suites if s.get("category") in selected]
        if not all_suites:
            logger.error("No suites matched categories: %s", ", ".join(sorted(selected)))
            sys.exit(1)

    text_suites = [s for s in all_suites if s.get("category") in text_categories]
    audio_suites = [s for s in all_suites if s.get("category") not in text_categories]
    needs_audio = bool(audio_suites)

    # --- WAV manifest check (audio suites only) ---
    wav_map = _load_manifest(args.questions, args.wav_dir)
    if needs_audio:
        missing = []
        for suite in audio_suites:
            for q in _iter_questions(suite):
                if q["id"] not in wav_map:
                    missing.append(q["id"])
        if missing:
            logger.error("Missing WAV files for: %s", ", ".join(missing))
            logger.error("Run prepare_audio.py first")
            sys.exit(1)

    # --- Shared module initialization ---
    language_code = "en-US"
    eval_db = str(output_dir / "eval.db")

    llm = OpenAILLM(
        model="gpt-5.4",
        temperature=0.7,
        reasoning_effort="none",
        max_tokens=256,
        tools=["web_search"],
    )
    storage = create_storage_backend("sqlite", db_path=eval_db)
    token_counter = create_token_counter(llm.model)
    tools_token_cost = get_tools_token_cost(llm.tools)

    embedder = create_embedder(expected_dimension=_DEFAULT_DIMENSION)
    memory_storage = SQLiteMemoryStorage(eval_db)
    trace_store = SQLiteTraceStore(eval_db)
    call_store = SQLiteCallStore(eval_db)
    embedder = TrackedEmbedder(embedder, call_store)
    retry_handler = OpenAIRetryHandler(call_store)
    logging.getLogger("openai._base_client").addHandler(retry_handler)
    vector_index = NumpyVectorIndex()

    # --- Text session factory ---
    def create_text_session(*, memory_enabled: bool = True, load_session_context: bool = True):
        from voice_pipeline.text_session import TextSession

        ms = memory_storage if memory_enabled else None
        retriever = MemoryRetriever(memory_storage, vector_index, embedder) if memory_enabled else None
        history = ConversationHistory(storage, token_counter)
        return TextSession(
            llm=llm,
            history=history,
            token_counter=token_counter,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            tools_token_cost=tools_token_cost,
            memory_storage=ms,
            retriever=retriever,
            load_session_context=load_session_context,
        )

    # --- Seed injection ---
    seed_session_map: dict[int, str] = {}
    seed_episode_map: dict[str, list[int]] = {}
    has_memory = any(s.get("category") == "memory" for s in all_suites)
    if has_memory and seed_data:
        logger.info("Injecting %d seed sessions", len(seed_data["sessions"]))
        seed_session_map = _inject_seeds(seed_data, memory_storage, vector_index, embedder, token_counter)
        seed_sids = list(seed_session_map.values())
        eps_by_session = memory_storage.get_episodes_by_session_ids(seed_sids)
        for sid, episodes in eps_by_session.items():
            seed_episode_map[sid] = [ep.id for ep in episodes]
        logger.info("Seed injection complete: %d episodes total", sum(len(v) for v in seed_episode_map.values()))

    # --- Audio module initialization (only if needed) ---
    asr = None
    raw_tts = None
    tts = None
    vap = None
    turngpt = None
    bridge = None
    led = None
    executor = None
    audio_queue = None
    audio_input = None
    prev_async: list[AsyncVAP | AsyncTurnGPT] = []

    if needs_audio:
        asr = GoogleCloudASR(language_code=language_code)
        raw_tts = OpenAITTS()
        tts = TrackedTTS(raw_tts, call_store)
        vap = MaAIVAPWrapper(raw_tts.output_sample_rate)
        turngpt = TurnGPTWrapper()
        silero_vad_model = load_silero_vad(onnx=True)
        _vad_buf = bytearray()
        _vad_last_score = [0.0]
        _vad_call_count = [0]
        _VAD_INFER_INTERVAL = 3  # 3프레임(90ms)마다 추론, 사이는 캐시 반환
        _SILERO_CHUNK_BYTES = 512 * 2  # 512 samples × 16-bit

        def vad_fn(frame: AudioFrame) -> float:
            _vad_buf.extend(frame)
            _vad_call_count[0] += 1
            if _vad_call_count[0] % _VAD_INFER_INTERVAL != 0:
                return _vad_last_score[0]
            while len(_vad_buf) >= _SILERO_CHUNK_BYTES:
                chunk = bytes(_vad_buf[:_SILERO_CHUNK_BYTES])
                del _vad_buf[:_SILERO_CHUNK_BYTES]
                samples = torch.frombuffer(bytearray(chunk), dtype=torch.int16).float() / 32768.0
                _vad_last_score[0] = silero_vad_model(samples, SAMPLE_RATE).item()
            return _vad_last_score[0]

        def reset_vad() -> None:
            # Silero LSTM 상태는 이전 오디오 이력을 유지하며 자연 회복되지 않음 —
            # 큰 발화 이력이 남으면 조용한 음성을 통째로 놓침. 세션 시작마다 초기화.
            silero_vad_model.reset_states()
            _vad_buf.clear()
            _vad_last_score[0] = 0.0
            _vad_call_count[0] = 0

        bridge = CppBridge()
        led = LEDController(enabled=False)
        executor = ThreadPoolExecutor(max_workers=SpeechGenerator.MAX_WORKERS)

        audio_queue = queue.Queue(maxsize=_AUDIO_QUEUE_SIZE)
        audio_input = AudioInput(audio_queue)

        if _asound is not None:
            _asound.snd_lib_error_set_handler(None)

    # --- Audio session factory ---
    shutdown_event = threading.Event()

    def create_session(
        *,
        on_turn_complete: Callable[[float], None] | None = None,
        on_turn_shift: Callable[[float, str], None] | None = None,
        on_playback_started: Callable[[], None] | None = None,
        on_generation_failed: Callable[[], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        memory_enabled: bool = True,
        skip_generation: bool = False,
        record_path: str | None = None,
    ) -> SessionComponents:
        for wrapper in prev_async:
            wrapper.stop()
        prev_async.clear()

        vap.reset()
        turngpt.reset()
        reset_vad()

        session_id = str(uuid.uuid4())
        tts.session_id = session_id
        retry_handler.session_id = session_id
        embedder.session_id = session_id
        async_vap = AsyncVAP(vap, call_store=call_store, session_id=session_id)
        async_turngpt = AsyncTurnGPT(turngpt, call_store=call_store, session_id=session_id)
        prev_async.extend([async_vap, async_turngpt])

        ms = memory_storage if memory_enabled else None
        history = ConversationHistory(storage, token_counter)
        retriever = MemoryRetriever(memory_storage, vector_index, embedder) if memory_enabled else None
        turn_detector = TurnDetector(
            async_vap,
            async_turngpt,
            embedder,
            vad_fn=vad_fn,
            vad_reset_fn=reset_vad,
            call_store=call_store,
            session_id=session_id,
        )
        generator = SpeechGenerator(
            llm,
            tts,
            history,
            token_counter,
            DEFAULT_SYSTEM_PROMPT,
            executor,
            tools_token_cost=tools_token_cost,
            memory_storage=ms,
            retriever=retriever,
            session_id=session_id,
        )
        session_loop = SessionLoop(
            asr=asr,
            turn_detector=turn_detector,
            speech_generator=generator,
            cpp_bridge=bridge,
            history=history,
            led=led,
            audio_queue=audio_queue,
            tts_sample_rate=tts.output_sample_rate,
            memory_storage=ms,
            session_id=session_id,
            token_counter=token_counter,
            trace_store=trace_store,
            shutdown_event=shutdown_event,
            on_turn_complete=on_turn_complete,
            on_turn_shift=on_turn_shift,
            on_playback_started=on_playback_started,
            on_generation_failed=on_generation_failed,
            on_cancel=on_cancel,
            disable_exit_keywords=True,
            skip_generation=skip_generation,
            record_path=record_path,
        )
        return SessionComponents(
            session_loop=session_loop,
            history=history,
            session_id=session_id,
        )

    # --- Signal handling & pause control ---
    def _handle_signal(*_: object) -> None:
        shutdown_event.set()

    signal.signal(signal.SIGINT, _handle_signal)

    pause_ctrl = _PauseController(shutdown_event)
    pause_ctrl.start()

    # --- Run eval ---
    session_map: list[dict] = []
    started_at = datetime.now().strftime(_TIMESTAMP_FORMAT)

    total_suites = len(text_suites) + len(audio_suites)
    logger.info("Eval starting — %d suites (%d text, %d audio)", total_suites, len(text_suites), len(audio_suites))

    # --- Run text suites ---
    for suite in text_suites:
        if pause_ctrl.wait_if_paused():
            break

        if suite.get("multi_turn"):
            scenarios = suite.get("scenarios", [])
            if args.quick and len(scenarios) > 1:
                scenarios = random.sample(scenarios, 1)
            logger.info(
                "Suite [text]: %s (%d/%d scenarios)",
                suite["name"],
                len(scenarios),
                len(suite.get("scenarios", [])),
            )
            for scenario in scenarios:
                if pause_ctrl.wait_if_paused():
                    break
                _run_text_multi_turn(
                    suite, scenario, session_map, create_text_session, seed_episode_map, seed_session_map
                )
        else:
            questions = suite.get("questions", [])
            if args.quick and len(questions) > 1:
                questions = random.sample(questions, 1)
            logger.info(
                "Suite [text]: %s (%d/%d questions)",
                suite["name"],
                len(questions),
                len(suite.get("questions", [])),
            )
            for question in questions:
                if pause_ctrl.wait_if_paused():
                    break
                _run_text_single_turn(
                    suite, question, session_map, create_text_session, seed_episode_map, seed_session_map
                )

    # --- Run audio suites ---
    if audio_suites and needs_audio:
        record_dir = str(output_dir / "recordings")
        Path(record_dir).mkdir(exist_ok=True)

        beep_wav = None
        if not args.no_beep:
            beep_wav = str(output_dir / "beep.wav")
            _make_beep_wav(beep_wav)
            logger.info("Session-start beep enabled: %s", beep_wav)
        player = QuestionPlayer(args.device, beep_wav=beep_wav)
        bridge.connect()
        audio_input.start()

        try:
            for suite in audio_suites:
                if pause_ctrl.wait_if_paused():
                    break

                if suite.get("multi_turn"):
                    scenarios = suite["scenarios"]
                    if args.quick and len(scenarios) > 1:
                        scenarios = random.sample(scenarios, 1)
                    logger.info(
                        "Suite: %s (%d/%d scenarios)",
                        suite["name"],
                        len(scenarios),
                        len(suite["scenarios"]),
                    )
                    for scenario in scenarios:
                        if pause_ctrl.wait_if_paused():
                            break
                        _run_multi_turn_suite(
                            suite,
                            scenario,
                            wav_map,
                            player,
                            session_map,
                            create_session,
                            audio_queue,
                            seed_session_map=seed_session_map,
                            seed_episode_map=seed_episode_map,
                            record_dir=record_dir,
                        )
                    continue

                questions = suite["questions"]
                if args.quick and len(questions) > 1:
                    questions = random.sample(questions, 1)

                logger.info("Suite: %s (%d/%d questions)", suite["name"], len(questions), len(suite["questions"]))

                if suite.get("category") == "interruption":
                    delays = suite.get("interrupt_delays_sec", [2.0])
                    interrupt_audios = suite.get("interrupt_audios", [])
                    if not interrupt_audios:
                        logger.error("Suite %s has no interrupt_audios", suite["name"])
                        continue
                    if args.quick:
                        delays = [delays[0], delays[-1]] if len(delays) > 1 else delays
                        interrupt_audios = random.sample(interrupt_audios, 1)
                    for delay in delays:
                        for int_id in interrupt_audios:
                            int_entry = wav_map.get(int_id, {})
                            interrupt_wav = int_entry.get("path", "")
                            if not interrupt_wav:
                                logger.error("No WAV for interrupt %s", int_id)
                                continue
                            for question in questions:
                                if pause_ctrl.wait_if_paused():
                                    break
                                q_entry = wav_map[question["id"]]
                                _run_interruption(
                                    suite,
                                    question,
                                    q_entry["path"],
                                    interrupt_wav,
                                    delay,
                                    player,
                                    session_map,
                                    create_session,
                                    audio_queue,
                                    interrupt_audio=int_id,
                                    seed_session_map=seed_session_map,
                                    seed_episode_map=seed_episode_map,
                                    record_dir=record_dir,
                                    voice=q_entry.get("voice", ""),
                                )
                else:
                    for question in questions:
                        if pause_ctrl.wait_if_paused():
                            break
                        q_entry = wav_map[question["id"]]
                        _run_single_turn(
                            suite,
                            question,
                            q_entry["path"],
                            player,
                            session_map,
                            create_session,
                            audio_queue,
                            seed_session_map=seed_session_map,
                            seed_episode_map=seed_episode_map,
                            record_dir=record_dir,
                            voice=q_entry.get("voice", ""),
                        )
        finally:
            audio_input.stop()
            bridge.disconnect()
            for wrapper in prev_async:
                wrapper.stop()
            prev_async.clear()
            executor.shutdown(wait=False)
            asr.stop()
            led.close()

    # --- Shared cleanup ---
    memory_storage.close()
    trace_store.close()
    call_store.close()

    # --- Save session mapping ---
    finished_at = datetime.now().strftime(_TIMESTAMP_FORMAT)
    successful = sum(1 for s in session_map if s["success"])

    pipeline_config: dict = {
        "llm_model": llm.model,
        "llm_temperature": llm.temperature,
        "writer_llm_model": "gpt-4o-mini",
        "tts_model": raw_tts._MODEL if raw_tts else None,
        "tts_voice": raw_tts._VOICE if raw_tts else None,
        "asr_model": asr._MODEL if asr else None,
        "asr_language": language_code,
        "vap_model": type(vap).__name__ if vap else None,
        "turngpt_model": type(turngpt).__name__ if turngpt else None,
        "vad_model": "silero_vad" if needs_audio else None,
    }
    runner_config: dict = {
        "quick": args.quick,
        "text": args.text,
        "category": args.category,
        "suites": [s["name"] for s in all_suites],
        "question_count": sum(len(list(_iter_questions(s))) for s in all_suites),
    }

    result = {
        "started_at": started_at,
        "finished_at": finished_at,
        "total": len(session_map),
        "successful": successful,
        "failed": len(session_map) - successful,
        "config": {
            "pipeline": pipeline_config,
            "runner": runner_config,
            "suite_descriptions": {s["name"]: s.get("description", "") for s in all_suites},
            "question_texts": {q["id"]: q["text"] for s in questions_data["suites"] for q in _iter_questions(s)},
        },
        "eval_db": eval_db,
        "seed_session_map": {str(k): v for k, v in seed_session_map.items()},
        "seed_file": str(seed_path) if seed_path else None,
        "sessions": session_map,
    }
    sessions_path = output_dir / "sessions.json"
    sessions_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    logger.info(
        "Eval complete: %d/%d successful — %s",
        successful,
        len(session_map),
        sessions_path,
    )

    # --- Report & Score & Dashboard ---
    from dashboard import build_html
    from report import build_report, print_summary
    from score import print_scores, score_report

    report = build_report(output_dir)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    scored = score_report(report)
    scored_path = output_dir / "scored.json"
    scored_path.write_text(json.dumps(scored, indent=2, ensure_ascii=False))

    dashboard_path = output_dir / "dashboard.html"
    dashboard_path.write_text(build_html(scored))

    print_summary(report)
    print_scores(scored)
    logger.info("Dashboard: %s", dashboard_path)


if __name__ == "__main__":
    main()
