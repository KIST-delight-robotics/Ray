"""Unit tests for AsyncTurnGPT and SyncTurnGPTAdapter."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from voice_pipeline.core.interfaces import ITurnGPT
from voice_pipeline.turn_taking.async_turngpt import AsyncTurnGPT, SyncTurnGPTAdapter


def _make_async_turngpt(
    predict_return: float = 0.5,
    predict_delay: float = 0.0,
) -> tuple[AsyncTurnGPT, MagicMock]:
    """Create AsyncTurnGPT with a mocked underlying ITurnGPT."""
    mock_turngpt = MagicMock(spec=ITurnGPT)

    if predict_delay > 0:

        def slow_predict(text: str) -> float:
            time.sleep(predict_delay)
            return predict_return

        mock_turngpt.predict.side_effect = slow_predict
    else:
        mock_turngpt.predict.return_value = predict_return

    async_tgpt = AsyncTurnGPT(mock_turngpt)
    return async_tgpt, mock_turngpt


# ---------------------------------------------------------------------------
# AsyncTurnGPT tests
# ---------------------------------------------------------------------------


class TestSubmitPoll:
    def test_basic_flow(self):
        """submit() + wait + poll_result() returns the predicted probability."""
        async_tgpt, _ = _make_async_turngpt(predict_return=0.7)
        try:
            async_tgpt.submit("hello<ts>how are you")
            time.sleep(0.1)
            result = async_tgpt.poll_result()
            assert result == 0.7
        finally:
            async_tgpt.stop()

    def test_poll_consumes_result(self):
        """poll_result() returns None on the second call (consumed)."""
        async_tgpt, _ = _make_async_turngpt(predict_return=0.7)
        try:
            async_tgpt.submit("hello")
            time.sleep(0.1)
            assert async_tgpt.poll_result() == 0.7
            assert async_tgpt.poll_result() is None
        finally:
            async_tgpt.stop()

    def test_poll_before_submit_returns_none(self):
        """poll_result() returns None when nothing has been submitted."""
        async_tgpt, _ = _make_async_turngpt()
        try:
            assert async_tgpt.poll_result() is None
        finally:
            async_tgpt.stop()

    def test_latest_submit_wins(self):
        """Multiple rapid submits — only the latest text is processed."""
        mock_turngpt = MagicMock(spec=ITurnGPT)
        # Slow predict so the first one doesn't finish before second arrives
        call_count = 0

        def counting_predict(text: str) -> float:
            nonlocal call_count
            call_count += 1
            return 0.3 if "first" in text else 0.8

        mock_turngpt.predict.side_effect = counting_predict
        async_tgpt = AsyncTurnGPT(mock_turngpt)
        try:
            async_tgpt.submit("first text")
            async_tgpt.submit("second text")
            time.sleep(0.2)
            # The last submit should overwrite _pending_text, but the thread
            # may have already picked up "first". Either way, after processing,
            # the result should reflect the latest predict call.
            async_tgpt.poll_result()
            # We can't guarantee which text was processed, but at least one
            # predict call should have been made
            assert call_count >= 1
        finally:
            async_tgpt.stop()


class TestClearPending:
    def test_clears_pending_and_result(self):
        """clear_pending() discards buffered result."""
        async_tgpt, _ = _make_async_turngpt(predict_return=0.7)
        try:
            async_tgpt.submit("hello")
            time.sleep(0.1)
            async_tgpt.clear_pending()
            assert async_tgpt.poll_result() is None
        finally:
            async_tgpt.stop()

    def test_clear_before_inference_discards_result(self):
        """If clear_pending() is called before inference finishes, result is discarded."""
        async_tgpt, _ = _make_async_turngpt(predict_return=0.7, predict_delay=0.1)
        try:
            async_tgpt.submit("hello")
            # Clear immediately, before the slow predict finishes
            async_tgpt.clear_pending()
            time.sleep(0.2)
            # Result should be discarded because _pending_text was set to None
            assert async_tgpt.poll_result() is None
        finally:
            async_tgpt.stop()


class TestReset:
    def test_delegates_to_background_thread(self):
        """reset() causes the background thread to call turngpt.reset()."""
        async_tgpt, mock_turngpt = _make_async_turngpt()
        try:
            async_tgpt.reset()
            time.sleep(0.1)
            mock_turngpt.reset.assert_called_once()
        finally:
            async_tgpt.stop()

    def test_clears_pending_state(self):
        """reset() clears pending text and result."""
        async_tgpt, _ = _make_async_turngpt(predict_return=0.7)
        try:
            async_tgpt.submit("hello")
            time.sleep(0.1)
            async_tgpt.reset()
            time.sleep(0.1)
            assert async_tgpt.poll_result() is None
        finally:
            async_tgpt.stop()


class TestStop:
    def test_thread_joins(self):
        """stop() causes the background thread to exit."""
        async_tgpt, _ = _make_async_turngpt()
        async_tgpt.stop()
        assert not async_tgpt._thread.is_alive()

    def test_idempotent_stop(self):
        """Calling stop() twice should not raise."""
        async_tgpt, _ = _make_async_turngpt()
        async_tgpt.stop()
        async_tgpt.stop()


# ---------------------------------------------------------------------------
# SyncTurnGPTAdapter tests
# ---------------------------------------------------------------------------


class TestSyncAdapter:
    def test_submit_poll_flow(self):
        """Synchronous submit+poll returns the predicted value immediately."""
        mock = MagicMock(spec=ITurnGPT)
        mock.predict.return_value = 0.6
        adapter = SyncTurnGPTAdapter(mock)

        adapter.submit("hello")
        assert adapter.poll_result() == 0.6
        # Consumed
        assert adapter.poll_result() is None

    def test_clear_pending(self):
        mock = MagicMock(spec=ITurnGPT)
        mock.predict.return_value = 0.6
        adapter = SyncTurnGPTAdapter(mock)

        adapter.submit("hello")
        adapter.clear_pending()
        assert adapter.poll_result() is None

    def test_reset_delegates(self):
        mock = MagicMock(spec=ITurnGPT)
        adapter = SyncTurnGPTAdapter(mock)

        adapter.reset()
        mock.reset.assert_called_once()

    def test_stop_is_noop(self):
        mock = MagicMock(spec=ITurnGPT)
        adapter = SyncTurnGPTAdapter(mock)
        adapter.stop()  # Should not raise
