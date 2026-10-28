"""Prompt templates and JSON schemas for memory write operations.

Contains:
- JSON schemas for structured output (episode extraction, user-notes update, dedup)
- Prompt builder functions for each LLM call in the write pipeline
"""

from __future__ import annotations

from typing import Any

from voice_pipeline.memory.storage import RADIO_ROLE
from voice_pipeline.memory.types import NOTE_KINDS, NoteKind, UserNote

# ---------------------------------------------------------------------------
# JSON schemas for structured output
# ---------------------------------------------------------------------------

EPISODE_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "name": "episode_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "episodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": ("Third-person narrative of the episode (1-3 sentences)."),
                        },
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["episodes"],
        "additionalProperties": False,
    },
}

USER_NOTES_UPDATE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "name": "user_notes_update",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "ops": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": ["add", "replace", "delete"]},
                        "list": {
                            "type": "string",
                            "enum": list(NOTE_KINDS),
                            "description": "List the item belongs to.",
                        },
                        "target": {
                            "type": ["string", "null"],
                            "description": "Id of the existing item (P1, R2, ...) for replace/delete. Null for add.",
                        },
                        "text": {
                            "type": ["string", "null"],
                            "description": "The full item sentence for add/replace. Null for delete.",
                        },
                    },
                    "required": ["op", "list", "target", "text"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["ops"],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_EPISODE_SYSTEM = """\
You are a memory extraction system. Given a conversation transcript between \
a user and an assistant, extract memorable episodes as third-person narratives.

## Rules
- Extract from USER utterances only. Do not extract facts from assistant responses.
- Each episode should be 1-3 sentences in third-person narrative form.
- Refer to the user as "the user"; do not guess their gender.
- Write in English. Keep names, titles and Korean-specific terms in the \
original language after the English, for example "plantain (질경이)".
- Preserve emotional context and situational details.
- Do NOT include dates or timestamps in the narrative text.
- Be selective: only extract episodes worth remembering. Return an empty array \
for trivial or meaningless conversations.
- If the user mentions a past event ("I went to Paris last week"), record that \
as a narrative about the user mentioning it, not as a direct fact.

## Radio talk
Lines marked "Assistant (radio talk)" are a radio talk the assistant hosts: \
prepared stories it reads aloud, chosen by the assistant, not by the user.
- When the user asks about or comments on the radio talk, record one episode \
that says so in context: the user was listening to the assistant's radio talk \
about the subject, and what they asked or said. Do not retell the talk's content, \
and do not describe the user's questions as a lasting interest.
- Brief reactions or controls alone (for example 음, 그렇구나, 계속해 줘, 잠깐만) \
are not worth an episode.
- What the user says about themselves during the radio talk (their own \
experiences, opinions, preferences) or about other topics is extracted as usual."""

_EPISODE_USER_TEMPLATE = """\
Session date: {session_date}

## Conversation
{transcript}"""


def build_episode_extraction_messages(
    utterances: list[tuple[str, str, str, int]],
    session_date: str,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Build messages for episode extraction LLM call.

    Args:
        utterances: List of (role, text, timestamp, token_count) tuples.
        session_date: Session date string for context.
        system_prompt: 추출 시스템 프롬프트 오버라이드 (중립 주입점 — 평가에서
            프롬프트 변형 실험용). ``None``이면 기본 프롬프트.

    Returns:
        Messages list with system + user message.
    """
    transcript = _format_transcript(utterances)
    return [
        {"role": "system", "content": system_prompt or _EPISODE_SYSTEM},
        {
            "role": "user",
            "content": _EPISODE_USER_TEMPLATE.format(
                session_date=session_date,
                transcript=transcript,
            ),
        },
    ]


_USER_NOTES_SYSTEM = """\
You maintain two short lists about the user of Ray, a desk robot that talks with \
them. Ray sees both lists at the start of every conversation. Given the current \
lists and one conversation transcript, return the changes the conversation calls for.

## Profile (P): who the user is
Identity facts the user stated plainly about themselves: their name or what to call \
them, job, workplace or school, major, where they live, family members, pets.
- Not states or events, however clearly stated: being busy this week, an upcoming \
exam, a cold, a trip.
- Not passing mentions: "work was tough today" tells that they work, not where.
- Not tastes, interests, opinions or personality.

## Preferences (R): how the user wants Ray to talk
Requests the user made to Ray about how Ray should speak or behave, marked as \
lasting ("from now on", "always", "앞으로", "항상"). Write each as an instruction \
to Ray, for example "Speak in 반말."
- Not requests for the moment: "louder", "tell me more", "shorter this time".
- Never inferred from the user's reactions, mood or laughter.

## Rules
- Use only what the user said about themselves or asked of Ray. Assistant lines, \
including "Assistant (radio talk)", are context, never evidence.
- One fact per item, as a full short sentence. Profile items start with "The user"; \
do not guess their gender.
- Write in English. Keep names and Korean terms in the original language after \
the English, for example "The user lives near Wolgok Station (월곡역)."
- add: new information. replace: the user corrected or changed an existing item; \
give its id and the full new sentence. delete: the user said an item is wrong or \
no longer true, or asked Ray to stop following a preference.
- If an item already says it, change nothing. Most conversations change nothing: \
return an empty list."""

_USER_NOTES_USER_TEMPLATE = """\
## Current lists
Profile:
{profile}

Preferences:
{preferences}

## Conversation
{transcript}"""

_NOTE_ID_PREFIX: dict[NoteKind, str] = {"profile": "P", "preference": "R"}


def label_user_notes(notes: list[UserNote]) -> dict[str, UserNote]:
    """Number notes per list in stored order (P1, P2, … / R1, …) — the ids the update call refers to."""
    labeled: dict[str, UserNote] = {}
    counts = dict.fromkeys(NOTE_KINDS, 0)
    for note in notes:
        counts[note.kind] += 1
        labeled[f"{_NOTE_ID_PREFIX[note.kind]}{counts[note.kind]}"] = note
    return labeled


def build_user_notes_messages(
    labeled_notes: dict[str, UserNote],
    utterances: list[tuple[str, str, str, int]],
) -> list[dict[str, Any]]:
    """Build messages for the user-notes update LLM call.

    Args:
        labeled_notes: Current notes keyed by their prompt id (:func:`label_user_notes`).
        utterances: Session transcript as (role, text, timestamp, token_count)
            tuples — the same input as episode extraction.

    Returns:
        Messages list with system + user message.
    """

    def section(kind: NoteKind) -> str:
        lines = [f"{label}. {note.text}" for label, note in labeled_notes.items() if note.kind == kind]
        return "\n".join(lines) or "(empty)"

    return [
        {"role": "system", "content": _USER_NOTES_SYSTEM},
        {
            "role": "user",
            "content": _USER_NOTES_USER_TEMPLATE.format(
                profile=section("profile"),
                preferences=section("preference"),
                transcript=_format_transcript(utterances),
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Episode deduplication across windows
# ---------------------------------------------------------------------------

EPISODE_DEDUP_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "name": "episode_dedup",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "MERGE",
                    "KEEP_BOTH",
                    "DISCARD_FIRST",
                    "DISCARD_SECOND",
                ],
            },
            "merged": {
                "type": ["string", "null"],
                "description": ("Merged narrative for MERGE action. Null for KEEP_BOTH / DISCARD_*."),
            },
        },
        "required": ["action", "merged"],
        "additionalProperties": False,
    },
}

_EPISODE_DEDUP_SYSTEM = """\
You are an episode deduplication system. You are given two episode \
descriptions extracted from overlapping windows of the same conversation. \
Determine whether they describe the same event and choose an action:

- MERGE: Same event described from different angles or with different \
details. Combine into a single third-person narrative (1-3 sentences) \
preserving all unique details from both. Return the merged text.
- KEEP_BOTH: Different events. Keep both episodes unchanged. Return null.
- DISCARD_FIRST: First episode is a strict subset of the second. \
Keep the second. Return null.
- DISCARD_SECOND: Second episode is a strict subset of the first. \
Keep the first. Return null."""

_EPISODE_DEDUP_USER_TEMPLATE = """\
A: {episode_a}
B: {episode_b}"""


def build_episode_dedup_messages(
    episode_a: str,
    episode_b: str,
) -> list[dict[str, Any]]:
    """Build messages for episode deduplication LLM call.

    Args:
        episode_a: Existing episode text.
        episode_b: Candidate episode text to compare.

    Returns:
        Messages list with system + user message.
    """
    return [
        {"role": "system", "content": _EPISODE_DEDUP_SYSTEM},
        {
            "role": "user",
            "content": _EPISODE_DEDUP_USER_TEMPLATE.format(episode_a=episode_a, episode_b=episode_b),
        },
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_transcript(utterances: list[tuple[str, str, str, int]]) -> str:
    """Format utterances into a readable conversation transcript."""
    lines: list[str] = []
    for role, text, timestamp, _token_count in utterances:
        label = "User" if role == "user" else "Assistant (radio talk)" if role == RADIO_ROLE else "Assistant"
        lines.append(f"[{timestamp}] {label}: {text}")
    return "\n".join(lines)
