from pathlib import Path
from unittest.mock import MagicMock

import pytest
import tiktoken

from voice_pipeline.adapters.gpt_live import LiveAppended, LiveClosed, LiveError, LiveFunctionCall, LiveResponseDone
from voice_pipeline.engines.gpt_live.story_append import StoryAppendTransfer, narration_text, script_parts


def count(text):
    return len(tiktoken.get_encoding("o200k_base").encode(text))


def transfer():
    live = MagicMock()
    call = LiveFunctionCall("call1", "start_story", "{}", "d1", "r1")
    result = StoryAppendTransfer(live, call, "snow", "첫 문장이에요. " * 150, count)
    result.begin(0)
    return result


def ack(t, now=1):
    event = LiveAppended(t.pending_kind, t.pending_id, 100, 100)
    t.handle(event, now)
    return event


def prepared():
    t = transfer()
    while t.state != "backend":
        ack(t)
    return t


def test_script_extraction_and_token_budget_preserve_original():
    body = Path("scripts/gpt_live/fixtures/snowflower_append.md").read_text()
    text = narration_text(body)
    assert "##" not in text and "play_song" not in text and "핵심 정서" not in text
    parts = script_parts(text, "snowflower", "x" * 32, count)
    assert len(parts) > 1 and all(count(p) <= 450 for p in parts)
    assert "".join(p.split("\n", 1)[1] for p in parts) == text
    assert text.endswith("한번 들어보실래요?")


def test_oversized_sentence_fails_before_sending():
    with pytest.raises(ValueError, match="sentence"):
        script_parts("눈송이" * 1000, "s", "t", count)


def test_ack_matching_and_backend_final_gate():
    t = transfer()
    original = t.pending_id
    t.handle(LiveAppended("thinking", original, 0, 0), 1)
    t.handle(LiveAppended("instructions", "wrong", 0, 0), 1)
    assert t.pending_id == original
    old_ack = ack(t)
    current = t.pending_id
    t.handle(old_ack, 1)
    assert t.pending_id == current
    while t.state != "backend":
        ack(t)
    assert t.live.submit_function_output.call_count == 1
    assert t.live.continue_response.call_count == 1
    assert not t.live.append_commentary.called
    for event in (
        LiveResponseDone("other", "r2", "completed", (), True),
        LiveResponseDone("d1", "r1", "completed", (), True),
        LiveResponseDone("d1", "r2", "completed", (), False),
    ):
        t.handle(event, 2)
    assert not t.live.append_commentary.called
    final = LiveResponseDone("d1", "r2", "completed", (), True)
    t.handle(final, 3)
    t.handle(final, 4)
    assert t.live.append_commentary.call_count == 1
    ack(t, 4)
    assert t.state == "started"


@pytest.mark.parametrize("mode", ["timeout", "error", "cancel", "closed"])
def test_failed_transfer_never_starts_on_late_events(mode):
    t = transfer()
    event = LiveAppended(t.pending_kind, t.pending_id, 0, 0)
    if mode == "timeout":
        t.tick(21)
    elif mode == "error":
        t.handle(LiveError("invalid", "too long", t.pending_id), 1)
    elif mode == "closed":
        t.handle(LiveClosed("lost"), 1)
    else:
        t.cancel(1)
    t.handle(event, 22)
    t.handle(LiveResponseDone("d1", "r2", "completed", (), True), 23)
    assert t.state in ("failed", "cancelled")
    assert not t.live.append_commentary.called
    if mode != "closed":
        assert '"error"' in t.live.submit_function_output.call_args.args[1]


@pytest.mark.parametrize("status,calls", [("failed", ()), ("incomplete", ())])
def test_backend_failure_or_new_tool_prevents_start(status, calls):
    t = prepared()
    t.handle(LiveResponseDone("d1", "r2", status, calls, True), 2)
    assert t.state == "failed"
    assert not t.live.append_commentary.called


def test_cancelled_transfer_ack_cannot_start_replacement():
    old = transfer()
    event = LiveAppended(old.pending_kind, old.pending_id, 0, 0)
    old.cancel(1)
    new = transfer()
    new.handle(event, 2)
    assert new.state == "preparing" and new.next_part == 0
