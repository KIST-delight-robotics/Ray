# Scratchpad

Claude's working memory. Free-form notes, observations, and context carried across sessions.


## TurnDetector Design Notes

TurnDetector is not yet implemented. These notes define the design requirements based on the reference paper and project-specific adaptation decisions. The paper's pseudo-code (Appendix A) is a reference, not a blueprint — the actual design must fit our decoupled architecture.

Reference paper: `docs/Applying General Turn-taking Models to Conversational Human-Robot Interaction.pdf` (Skantze & Irfan, 2025).


### Internal Turn State

TurnDetector tracks whose turn it is with two internal states: `USER_TURN` and `ROBOT_TURN`. This determines which algorithms run each frame.

- `USER_TURN`: VAP + TurnGPT + timing heuristics active. Can emit turn_shift and prepare.
- `ROBOT_TURN`: Interrupt detection only. TurnGPT is not invoked. Cannot emit turn_shift or prepare.

**Transitions:**
- `USER_TURN → ROBOT_TURN`: Immediately when turn_shift is emitted. TurnDetector resets per-frame state (prev_asr_text, timers, etc.) and transitions.
- `ROBOT_TURN → USER_TURN`: When Orchestrator calls reset() (after playback completes or interrupt is handled).

Note: `ROBOT_TURN` does not mean the robot is audibly speaking. It means the turn has logically shifted to the robot. There may be a gap before actual playback starts (response generation time). During this gap, robot_audio is None.

**Why internal transition on turn_shift (not external control):**
If turn state were only controlled externally (e.g., by robot_audio presence), there would be a timing gap between turn_shift and playback start. During this gap, user speech would enter the prepare path, which has trigger conditions (TurnGPT > 0.2 or 200ms elapsed). If the response completes before prepare fires, the Orchestrator would start playback while the user is speaking. With internal transition, user speech during the gap immediately triggers interrupt, avoiding this race.


### Interrupt Detection (ROBOT_TURN only)

Two sub-cases based on whether robot_audio is being provided:

**With robot_audio (robot is audibly speaking):**
VAP processes both channels. Distinguish genuine interrupt from backchannel:
- **Interrupt**: Both p_now AND p_fut favor the user → emit interrupt.
- **Backchannel** ("yeah", "mhm"): p_now may favor user but p_fut favors robot → no action, robot continues.

**Without robot_audio (gap before playback):**
VAP only has user audio, so backchannel distinction is not possible. But it's also not needed — the robot is not audibly speaking, so backchannels don't apply. Any user speech (`user_is_speaking`) → emit interrupt immediately.

**Empty robot turn after interrupt:** If interrupt results in empty robot text (nothing was spoken, or truncation yields empty text), the robot turn is not recorded in ConversationHistory. Consecutive user turns in history naturally represent "user continued speaking."


### Turn-Shift: Two Independent OR Paths (USER_TURN only)

The paper's core contribution for turn-shift detection. Both paths check when the user is not speaking. Either path alone can trigger turn_shift.

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


### Prepare: Speculative Response Generation (USER_TURN only)

TurnDetector decides when to trigger background LLM+TTS preparation.

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
