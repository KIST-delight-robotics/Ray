# Scratchpad

Claude's working memory. Free-form notes, observations, and context carried across sessions.


## TurnDetector Design Notes

TurnDetector is not yet implemented. These notes define the design requirements based on the reference paper and project-specific adaptation decisions. The paper's pseudo-code (Appendix A) is a reference, not a blueprint — the actual design must fit our decoupled architecture.

Reference paper: `docs/Applying General Turn-taking Models to Conversational Human-Robot Interaction.pdf` (Skantze & Irfan, 2025).


### Internal Turn State

TurnDetector must track whose turn it is internally. The paper's pseudo-code doesn't do this because its main loop has direct access to `Speech_Generator.is_speaking()` / `is_ready()`. Our TurnDetector is decoupled from SpeechGenerator and Orchestrator, so it needs its own state.

**Two states:**
- `LISTENING` — User's turn. Prepare and turn_shift logic active.
- `ROBOT_SPEAKING` — Robot's turn. Only interrupt logic active. Prepare and turn_shift must not run.

**Transitions:**
- `LISTENING → ROBOT_SPEAKING`: robot_audio starts being provided to process_frame.
- `ROBOT_SPEAKING → LISTENING`: robot_audio stops being provided.

Robot speaking state is determined by actual C++ bridge playback signals, not by Python-side intent. The Orchestrator provides robot_audio to TurnDetector only while C++ is actually playing back. This means the transition to ROBOT_SPEAKING only happens when playback truly begins.

**turn_shift does not change internal state.** After emitting turn_shift, TurnDetector resets per-frame state and stays in LISTENING. Only when robot_audio actually arrives does the state become ROBOT_SPEAKING.

Without this state tracking, problems arise: during robot playback with user silent, silence counters would accumulate and eventually trigger a spurious turn_shift. During ROBOT_SPEAKING, these checks simply don't run.


### Turn-Shift: Two Independent OR Paths

The paper's core contribution for turn-shift detection. Both paths run during LISTENING when the user is not speaking. Either path alone can trigger turn_shift.

**Path 1 — VAP (fast path):**
p_now and p_fut both favor the robot, sustained for ≥ MIN_GAP_TIME (500ms). If either value stops favoring the robot, the timer resets. This enables response times as low as 500ms.

**Path 2 — TurnGPT graduated timeout (fallback):**
Silence duration since user stopped speaking, compared against a timeout that varies by TurnGPT's turn completion probability:
- prob ≥ 0.3 → 500ms
- prob ≥ 0.2 → 1000ms
- prob ≥ 0.1 → 2000ms
- prob ≥ 0.0 → 3000ms (maximum, safety fallback)

This ensures the robot eventually takes the turn even if VAP fails to detect yield, while still being fast when TurnGPT is confident.

**Graduated timeout and ASR timing:** Every ASR text change triggers TurnGPT re-evaluation, which updates the probability and recalculates the timeout. By the time silence truly starts (no more ASR changes), the timeout already reflects the most accurate TurnGPT probability. No special synchronization needed.


### Interrupt vs Backchannel Distinction

During ROBOT_SPEAKING, when user speech is detected (VAP's user_is_speaking), the system must distinguish genuine interrupts from backchannels ("yeah", "mhm").

**Interrupt** (robot should stop): Both p_now AND p_fut favor the user.
**Backchannel** (robot continues): p_now may favor user but p_fut favors robot, indicating brief user speech. Robot should not stop.

Only genuine interrupts emit the interrupt signal.

**Empty robot turn after interrupt:** If an interrupt occurs very early in playback (or if truncation results in empty text), the robot turn is simply not recorded in ConversationHistory. The result is consecutive user turns in history, which is a natural representation of "user continued speaking."


### Prepare: Speculative Response Generation

During LISTENING, TurnDetector decides when to trigger background LLM+TTS preparation.

**Trigger condition (from paper):** ASR text has changed AND either:
- TurnGPT probability > threshold (0.2), OR
- Time since last ASR change ≥ timeout (200ms)

**Gated by similarity:** Compare current ASR text against the text at the time of the last prepare emit. If similarity is above threshold (0.8), skip — the prepared response is still likely valid. This avoids redundant LLM+TTS work.

Prepare can fire multiple times as text evolves. Each prepare cancels the previous generation and starts fresh (this cancellation is Orchestrator's responsibility, not TurnDetector's).


### VAP Output Convention

VAPResult contains p_now, p_fut, and user_is_speaking. The p_now/p_fut values come from the VAP model's output at a specific channel index.

The exact index choice (0 = user probability, 1 = robot probability) doesn't matter as long as:
1. VAPResult docstring clearly defines the semantic meaning of the values.
2. TurnDetector threshold comparisons are consistent with that definition.
3. The convention is documented in one place and followed everywhere.

Whatever convention is chosen, the paper's threshold values (0.4, 0.5) need to be adapted to match. If p_now means "probability robot should speak," use the paper's thresholds directly. If it means "probability user should speak," flip the comparisons.


### Preserved Components

The following are already implemented and verified correct:
- `ITurnDetector` interface: `process_frame()`, `notify_turn_complete()`, `reset()`
- `TurnDecision` type: turn_shift, interrupt, prepare (at most one True per frame)
- `VAPResult` type: p_now, p_fut, user_is_speaking
- VAP wrapper (`vap.py`): stereo buffer, rolling window, periodic inference
- TurnGPT wrapper (`turngpt.py`): `<ts>`-delimited dialog, KV cache, context window eviction
- `notify_turn_complete()` builds dialog context for TurnGPT across turns
- `TurnDetectorConfig` placeholder exists — fields to be defined with the implementation
- `TurnDetectorError` exception class
