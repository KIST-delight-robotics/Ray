# Scratchpad

Claude's working memory. Free-form notes, observations, and context carried across sessions.


## TurnDetector Redesign Notes

Previous implementation was removed because it did not follow the paper's algorithm.
Reference paper: `docs/Applying General Turn-taking Models to Conversational Human-Robot Interaction.pdf` (Skantze & Irfan, 2025).
These notes capture design discussion conclusions that cannot be derived from the paper or architecture docs alone.


### Why the previous implementation was wrong

The old turn_detector.py had these critical problems:
1. **Interrupt**: Treated ALL user speech during robot playback as interrupt. The paper uses VAP p_now/p_fut thresholds to distinguish genuine interrupts from backchannels (e.g., "yeah", "mhm"). Only when both p_now AND p_fut favor the user is it a real interrupt.
2. **Turn-shift**: Used pure silence frame counting with no VAP involvement. The paper has two independent OR paths: (a) VAP path (p_now/p_fut favor robot for ≥ 500ms), (b) TurnGPT graduated timeout (silence ≥ timeout that varies by TurnGPT probability: 0.3→500ms, 0.2→1000ms, 0.1→2000ms, 0.0→3000ms).
3. **Prepare**: Required 800ms stability AND TurnGPT threshold. The paper uses TurnGPT OR 200ms timeout, gated by semantic similarity to avoid redundant generation.
4. **VAP p_now/p_fut were never used** in turn-shift or interrupt decisions — only `user_is_speaking` was used.


### Internal turn state tracking (not in paper)

The paper's pseudo-code branches only on `user_is_speaking` (voice activity) and gates actions with `Speech_Generator.is_speaking()` / `Speech_Generator.is_ready()`. Our TurnDetector is decoupled from SpeechGenerator, so it needs its own state tracking.

**Two internal states:**
- `LISTENING` — User's turn. Run prepare/turn_shift logic.
- `ROBOT_SPEAKING` — Robot's turn. Only run interrupt logic.

**Transitions:**
- `LISTENING → ROBOT_SPEAKING`: When robot_audio starts being provided (= CppBridge confirmed playback, Orchestrator feeds robot audio frames).
- `ROBOT_SPEAKING → LISTENING`: When robot_audio stops being provided (playback ended or interrupted).

**Key consequence — turn_shift does NOT change internal state:**
After TurnDetector emits turn_shift, it calls reset() and stays in LISTENING. The robot hasn't started speaking yet (C++ bridge hasn't confirmed playback). Only when Orchestrator begins providing robot_audio does the state change to ROBOT_SPEAKING.

**Why this matters — the gap between turn_shift and playback start:**
If the user speaks again during this gap (turn_shift emitted, but robot not yet playing):
- TurnDetector is still LISTENING → emits prepare or turn_shift normally (NOT interrupt)
- Orchestrator handles this: combines saved user message + new ASR text, cancels/restarts generation
- This is more natural than treating it as interrupt, because (a) robot said nothing yet so there's nothing to truncate, and (b) the user is just continuing their thought
- ARCHITECTURE.md's `awaiting_response` section already describes this flow

**Robot speaking state must be based on C++ bridge signals** (actual playback), not on Python-side "I sent audio to the bridge" events. The Orchestrator determines this from CppBridge events and communicates it to TurnDetector by providing/not providing robot_audio frames.


### VAP output convention

VAPResult.p_now and p_fut values: the exact channel index (0 or 1) doesn't matter as long as:
1. The VAPResult docstring clearly defines what the values mean
2. The TurnDetector applies thresholds consistently with that definition

Current vap.py uses index 0. Whichever convention is chosen, document it clearly in VAPResult and use matching threshold directions in TurnDetector. The paper's pseudo-code uses a convention where high p_now = robot should speak, but the code can use the opposite as long as comparisons are flipped accordingly.


### Graduated timeout and ASR timing

The TurnGPT graduated timeout naturally handles ASR timing concerns: every ASR text change triggers TurnGPT re-evaluation, which updates the probability, which recalculates the timeout. By the time silence truly starts (no more ASR changes), TurnGPT has already processed the final text and the timeout reflects the most accurate probability. No special synchronization needed.


### Prepare signal design details

- Similarity comparison target: "the ASR text at the time of the last prepare emit" (not the last text change). This avoids re-generating when the text hasn't meaningfully changed since the last preparation.
- Paper uses sentence embedding similarity (all-MiniLM-L6-v2). We may use simpler methods (e.g., SequenceMatcher) initially if embedding adds too much latency. The key behavior is: avoid redundant LLM+TTS generation for semantically equivalent inputs.
- `prepare` can fire multiple times as text evolves — each one cancels the previous preparation and starts fresh.


### What's preserved from old implementation

- `ITurnDetector` interface (process_frame, notify_turn_complete, reset) — still valid
- `TurnDecision` type (turn_shift, interrupt, prepare) — still valid
- `VAPResult` type — still valid
- VAP wrapper (vap.py) — verified correct, no changes needed
- TurnGPT wrapper (turngpt.py) — verified correct, no changes needed
- `TurnDetectorConfig` placeholder in config.py — fields to be redefined
- `TurnDetectorError` exception — still valid
- `notify_turn_complete` for building TurnGPT dialog context — good design, keep


### ITurnDetector interface considerations for redesign

Current interface signature:
```python
def process_frame(self, user_audio: AudioFrame, asr_text: str, robot_audio: AudioFrame | None = None) -> TurnDecision
def notify_turn_complete(self, role: Literal["user", "robot"], text: str) -> None
def reset(self) -> None
```

This should remain largely unchanged. The robot_audio parameter serves double duty: feeds VAP's robot channel AND signals that the robot is currently speaking (ROBOT_SPEAKING state). This is fine because Orchestrator already only provides robot_audio during actual C++ playback.
