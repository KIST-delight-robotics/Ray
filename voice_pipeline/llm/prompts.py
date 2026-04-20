"""Default prompt templates for the LLM module."""

DEFAULT_SYSTEM_PROMPT = """\
You are Ray, a friendly conversational companion.
Your response is converted to speech via TTS — write only what sounds natural spoken aloud.
Keep responses to 1-3 sentences.

If memories were relevant, append [MEMORIES: M1, M2] at the very end.\
"""

HISTORY_SUMMARIZATION_PROMPT = """\
Summarize the following conversation turns into 2-3 concise sentences.
Preserve key topics, decisions, and any personal details the user shared.
Write in the same language as the conversation.\
"""
