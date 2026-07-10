"""Prompt templates and JSON schemas for memory write operations.

Contains:
- Profile schema definition (data-driven topic/subtopic structure)
- JSON schemas for structured output (episode extraction, profile extraction, merge)
- Prompt builder functions for each LLM call in the write pipeline
"""

from __future__ import annotations

from typing import Any

from voice_pipeline.core.types import TokenCounter
from voice_pipeline.memory.types import Episode, Profile

# ---------------------------------------------------------------------------
# Profile schema
# ---------------------------------------------------------------------------

PROFILE_SCHEMA: dict[str, list[str]] = {
    "basic_info": ["name", "age", "location", "occupation", "language"],
    "interest": ["movie", "music", "book", "game", "food", "sport", "hobby"],
    "personality": ["traits", "values", "communication_style"],
    "interaction_style": ["tone_preference", "topic_preference", "humor_style"],
}


def format_profile_schema() -> str:
    """Format the profile schema as a human-readable block for prompt injection."""
    lines: list[str] = []
    for topic, subtopics in PROFILE_SCHEMA.items():
        lines.append(f"- {topic}")
        for sub in subtopics:
            lines.append(f"  - {sub}")
    return "\n".join(lines)


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

PROFILE_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "name": "profile_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "sub_topic": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["topic", "sub_topic", "content"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["facts"],
        "additionalProperties": False,
    },
}

PROFILE_MERGE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "name": "profile_merge",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["APPEND", "UPDATE", "ABORT"],
                        },
                        "fact_index": {
                            "type": "integer",
                            "description": "1-based index of the new fact (F1, F2, ...).",
                        },
                        "new_content": {
                            "type": ["string", "null"],
                            "description": ("Merged content for UPDATE, or new content for APPEND. Null for ABORT."),
                        },
                    },
                    "required": ["action", "fact_index", "new_content"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["actions"],
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
- Preserve emotional context and situational details.
- Do NOT include dates or timestamps in the narrative text.
- Be selective: only extract episodes worth remembering. Return an empty array \
for trivial or meaningless conversations.
- If the user mentions a past event ("I went to Paris last week"), record that \
as a narrative about the user mentioning it, not as a direct fact."""

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


_PROFILE_EXTRACTION_SYSTEM = """\
You are a profile extraction system. Given a list of episode summaries from \
a conversation, extract user profile facts that match the schema categories below.

## Profile Schema
{schema}

## Rules
- Only extract facts about the USER, not the assistant.
- Map each fact to a topic and sub_topic from the schema.
- You may create new sub_topics within existing topics if the schema \
does not cover the specific attribute.
- Keep each fact concise (one sentence or phrase).
- Only extract when there is clear evidence, not speculation.
- Return an empty array if no profile-relevant information is found."""

_PROFILE_EXTRACTION_USER_TEMPLATE = """\
## Episodes
{episodes}"""


def build_profile_extraction_messages(
    episodes: list[Episode],
    profile_schema_text: str,
) -> list[dict[str, Any]]:
    """Build messages for profile fact extraction LLM call.

    Args:
        episodes: Extracted episodes to analyze for profile facts.
        profile_schema_text: Formatted profile schema string.

    Returns:
        Messages list with system + user message.
    """
    episode_text = "\n".join(f"- {ep.text}" for ep in episodes)
    return [
        {
            "role": "system",
            "content": _PROFILE_EXTRACTION_SYSTEM.format(schema=profile_schema_text),
        },
        {
            "role": "user",
            "content": _PROFILE_EXTRACTION_USER_TEMPLATE.format(episodes=episode_text),
        },
    ]


_PROFILE_MERGE_SYSTEM = """\
You are a profile merge system. Each fact below shows new information and the \
current content of the matching profile slot (if any). For each fact, decide:

- APPEND: No existing slot matches. The fact will be added as a new slot.
- UPDATE: The fact updates or extends the existing slot. Provide the merged \
content that combines old and new information.
- ABORT: The fact is redundant (already captured) or too trivial to store.

Return one action per fact."""

_PROFILE_MERGE_USER_TEMPLATE = """\
## Facts
{facts}"""


def build_profile_merge_messages(
    existing_profiles: list[Profile],
    new_facts: list[dict[str, str]],
    max_content_tokens: int,
    warn_ratio: float,
    token_counter: TokenCounter | None = None,
) -> list[dict[str, Any]]:
    """Build messages for profile merge LLM call.

    Each fact is presented with its matching existing slot content (by
    topic::sub_topic key), so the LLM does not need a separate slot list.
    Slots near the token limit get an inline warning.

    Args:
        existing_profiles: Current profile slots.
        new_facts: Extracted facts to merge.
        max_content_tokens: Max tokens per slot content.
        warn_ratio: content 토큰이 예산의 몇 배를 넘으면 요약 경고 표시 (0.0–1.0).
        token_counter: Optional token counter for measuring slot sizes.

    Returns:
        Messages list with system + user message.
    """
    # Build (topic, sub_topic) → Profile lookup
    slot_map: dict[tuple[str, str], Profile] = {}
    for p in existing_profiles:
        slot_map[(p.topic, p.sub_topic)] = p

    fact_lines: list[str] = []
    for i, f in enumerate(new_facts, 1):
        key = (f["topic"], f["sub_topic"])
        existing = slot_map.get(key)

        line = f"F{i}. {f['topic']}::{f['sub_topic']} — {f['content']}"
        if existing is None:
            line += "\n    Current: (none)"
        else:
            current_tokens = token_counter(existing.content) if token_counter else None
            line += f"\n    Current: {existing.content}"
            if current_tokens is not None and current_tokens > max_content_tokens * warn_ratio:
                line += (
                    f"\n    ⚠ Near token limit ({current_tokens}/{max_content_tokens})."
                    " If updating, summarize to stay under limit."
                )
        fact_lines.append(line)

    facts_text = "\n".join(fact_lines)

    return [
        {"role": "system", "content": _PROFILE_MERGE_SYSTEM},
        {
            "role": "user",
            "content": _PROFILE_MERGE_USER_TEMPLATE.format(facts=facts_text),
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
        label = "User" if role == "user" else "Assistant"
        lines.append(f"[{timestamp}] {label}: {text}")
    return "\n".join(lines)
