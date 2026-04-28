"""Tests for voice_pipeline.asr.asr (GoogleCloudASR)."""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from unittest.mock import MagicMock, patch

import pytest

from voice_pipeline.asr.asr import GoogleCloudASR
from voice_pipeline.asr.exceptions import ASRError

_SENTINEL = GoogleCloudASR._SENTINEL

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_asr(language_code: str = "en-US") -> GoogleCloudASR:
    return GoogleCloudASR(language_code=language_code)


def _make_response(transcript: str, *, is_final: bool = False) -> MagicMock:
    """Build a mock StreamingRecognizeResponse with one result."""
    alternative = MagicMock()
    alternative.transcript = transcript

    result = MagicMock()
    result.alternatives = [alternative]
    result.is_final = is_final

    response = MagicMock()
    response.results = [result]
    return response


def _make_multi_result_response(
    *parts: tuple[str, bool],
) -> MagicMock:
    """Build a mock response with multiple results.

    Each *part* is ``(transcript, is_final)``.  This mirrors Google STT
    behaviour where a single response contains a stable prefix + speculative
    suffix, or a final + following interim.
    """
    results = []
    for transcript, is_final in parts:
        alt = MagicMock()
        alt.transcript = transcript
        result = MagicMock()
        result.alternatives = [alt]
        result.is_final = is_final
        results.append(result)

    response = MagicMock()
    response.results = results
    return response


def _make_empty_response() -> MagicMock:
    """Build a mock response with no alternatives."""
    result = MagicMock()
    result.alternatives = []

    response = MagicMock()
    response.results = [result]
    return response


# ---------------------------------------------------------------------------
# Tests — idle state
# ---------------------------------------------------------------------------


class TestIdleState:
    def test_get_text_before_start(self) -> None:
        asr = _make_asr()
        assert asr.get_text() == ""

    def test_feed_audio_before_start(self, caplog: pytest.LogCaptureFixture) -> None:
        asr = _make_asr()
        with caplog.at_level(logging.WARNING, logger="voice_pipeline.asr"):
            asr.feed_audio(b"\x00" * 960)
        assert "not running" in caplog.text


# ---------------------------------------------------------------------------
# Tests — start / stop lifecycle
# ---------------------------------------------------------------------------


@patch("voice_pipeline.asr.asr.speech.SpeechClient")
class TestStartStop:
    def test_start_creates_client_and_thread(self, mock_client_cls: MagicMock) -> None:
        block = threading.Event()

        def fake_streaming_recognize(config, requests):
            # Block until stop() sends sentinel, keeping the thread alive
            block.wait(timeout=10.0)
            return iter([])

        mock_client = MagicMock()
        mock_client.streaming_recognize.side_effect = fake_streaming_recognize
        mock_client_cls.return_value = mock_client

        asr = _make_asr()
        asr.start()
        try:
            mock_client_cls.assert_called_once()
            assert asr._running.is_set()
            assert asr._reader_thread is not None
            assert asr._reader_thread.is_alive()
        finally:
            block.set()
            asr.stop()

    def test_start_while_running_is_noop(self, mock_client_cls: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
        mock_client = MagicMock()
        mock_client.streaming_recognize.return_value = iter([])
        mock_client_cls.return_value = mock_client

        asr = _make_asr()
        asr.start()
        try:
            with caplog.at_level(logging.WARNING, logger="voice_pipeline.asr"):
                asr.start()
            assert "already running" in caplog.text
            assert mock_client_cls.call_count == 1
        finally:
            asr.stop()

    def test_stop_cleans_up(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.streaming_recognize.return_value = iter([])
        mock_client_cls.return_value = mock_client

        asr = _make_asr()
        asr.start()
        asr.stop()

        assert not asr._running.is_set()
        assert asr._client is None
        assert asr._audio_queue is None
        mock_client.transport.close.assert_called_once()

    def test_stop_when_not_running(self, mock_client_cls: MagicMock) -> None:
        asr = _make_asr()
        asr.stop()  # Should not raise


# ---------------------------------------------------------------------------
# Tests — feed + get transcript
# ---------------------------------------------------------------------------


@patch("voice_pipeline.asr.asr.speech.SpeechClient")
class TestTranscription:
    def test_feed_and_get_interim_transcript(self, mock_client_cls: MagicMock) -> None:
        transcript_ready = threading.Event()
        interim_response = _make_response("안녕하세요")

        def fake_streaming_recognize(config, requests):
            # Consume one request to simulate receiving audio
            for _ in requests:
                break
            transcript_ready.set()
            return iter([interim_response])

        mock_client = MagicMock()
        mock_client.streaming_recognize.side_effect = fake_streaming_recognize
        mock_client_cls.return_value = mock_client

        asr = _make_asr()
        asr.start()
        try:
            asr.feed_audio(b"\x00" * 960)
            # Wait for the reader thread to process the response
            assert transcript_ready.wait(timeout=5.0)
            # Give a small window for the lock-protected write
            _wait_for_transcript(asr, "안녕하세요")
            assert asr.get_text() == "안녕하세요"
        finally:
            asr.stop()

    def test_feed_and_get_final_transcript(self, mock_client_cls: MagicMock) -> None:
        transcript_ready = threading.Event()
        final_response = _make_response("반갑습니다", is_final=True)

        def fake_streaming_recognize(config, requests):
            for _ in requests:
                break
            transcript_ready.set()
            return iter([final_response])

        mock_client = MagicMock()
        mock_client.streaming_recognize.side_effect = fake_streaming_recognize
        mock_client_cls.return_value = mock_client

        asr = _make_asr()
        asr.start()
        try:
            asr.feed_audio(b"\x00" * 960)
            assert transcript_ready.wait(timeout=5.0)
            _wait_for_transcript(asr, "반갑습니다")
            assert asr.get_text() == "반갑습니다"
        finally:
            asr.stop()

    def test_transcript_updates_to_latest(self, mock_client_cls: MagicMock) -> None:
        all_consumed = threading.Event()
        r1 = _make_response("안")
        r2 = _make_response("안녕")
        r3 = _make_response("안녕하세요", is_final=True)

        def fake_streaming_recognize(config, requests):
            for _ in requests:
                break
            all_consumed.set()
            return iter([r1, r2, r3])

        mock_client = MagicMock()
        mock_client.streaming_recognize.side_effect = fake_streaming_recognize
        mock_client_cls.return_value = mock_client

        asr = _make_asr()
        asr.start()
        try:
            asr.feed_audio(b"\x00" * 960)
            assert all_consumed.wait(timeout=5.0)
            _wait_for_transcript(asr, "안녕하세요")
            assert asr.get_text() == "안녕하세요"
        finally:
            asr.stop()

    def test_multiple_finals_accumulated(self, mock_client_cls: MagicMock) -> None:
        """Multiple is_final results within one stream are concatenated."""
        done = threading.Event()
        r1 = _make_response("안녕하세요, ", is_final=True)
        r2 = _make_response("오늘 날씨 어때요?", is_final=True)

        def fake_streaming_recognize(config, requests):
            for _ in requests:
                break
            done.set()
            return iter([r1, r2])

        mock_client = MagicMock()
        mock_client.streaming_recognize.side_effect = fake_streaming_recognize
        mock_client_cls.return_value = mock_client

        asr = _make_asr()
        asr.start()
        try:
            asr.feed_audio(b"\x00" * 960)
            assert done.wait(timeout=5.0)
            _wait_for_transcript(asr, "안녕하세요, 오늘 날씨 어때요?")
            assert asr.get_text() == "안녕하세요, 오늘 날씨 어때요?"
        finally:
            asr.stop()

    def test_interim_appended_after_final(self, mock_client_cls: MagicMock) -> None:
        """Interim result is appended after accumulated finals."""
        done = threading.Event()
        r1 = _make_response("첫번째. ", is_final=True)
        r2 = _make_response("두번째 진행중")  # interim

        def fake_streaming_recognize(config, requests):
            for _ in requests:
                break
            done.set()
            return iter([r1, r2])

        mock_client = MagicMock()
        mock_client.streaming_recognize.side_effect = fake_streaming_recognize
        mock_client_cls.return_value = mock_client

        asr = _make_asr()
        asr.start()
        try:
            asr.feed_audio(b"\x00" * 960)
            assert done.wait(timeout=5.0)
            _wait_for_transcript(asr, "첫번째. 두번째 진행중")
            assert asr.get_text() == "첫번째. 두번째 진행중"
        finally:
            asr.stop()

    def test_interim_replaced_by_next_interim(self, mock_client_cls: MagicMock) -> None:
        """Successive interim results replace each other, not accumulate."""
        done = threading.Event()
        r1 = _make_response("가")  # interim
        r2 = _make_response("가나")  # interim (replaces previous)
        r3 = _make_response("가나다", is_final=True)

        def fake_streaming_recognize(config, requests):
            for _ in requests:
                break
            done.set()
            return iter([r1, r2, r3])

        mock_client = MagicMock()
        mock_client.streaming_recognize.side_effect = fake_streaming_recognize
        mock_client_cls.return_value = mock_client

        asr = _make_asr()
        asr.start()
        try:
            asr.feed_audio(b"\x00" * 960)
            assert done.wait(timeout=5.0)
            _wait_for_transcript(asr, "가나다")
            assert asr.get_text() == "가나다"
            # Verify interim is cleared after final
            with asr._lock:
                assert asr._interim_transcript == ""
                assert asr._final_transcript == "가나다"
        finally:
            asr.stop()

    def test_reset_clears_accumulated_finals(self, mock_client_cls: MagicMock) -> None:
        """reset() clears both accumulated finals and interim."""
        done = threading.Event()
        r1 = _make_response("첫번째. ", is_final=True)
        r2 = _make_response("두번째. ", is_final=True)

        call_count = 0

        def fake_streaming_recognize(config, requests):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                for _ in requests:
                    break
                done.set()
                return iter([r1, r2])
            return iter([])

        mock_client = MagicMock()
        mock_client.streaming_recognize.side_effect = fake_streaming_recognize
        mock_client_cls.return_value = mock_client

        asr = _make_asr()
        asr.start()
        try:
            asr.feed_audio(b"\x00" * 960)
            assert done.wait(timeout=5.0)
            _wait_for_transcript(asr, "첫번째. 두번째. ")
            assert asr.get_text() == "첫번째. 두번째. "

            asr.reset()
            assert asr.get_text() == ""
            with asr._lock:
                assert asr._final_transcript == ""
                assert asr._interim_transcript == ""
        finally:
            asr.stop()

    def test_multi_interim_response_concatenated(self, mock_client_cls: MagicMock) -> None:
        """Multiple interims in one response are concatenated, not overwritten.

        Google STT may split a single response into a stable prefix and a
        speculative suffix.  get_text() must return them joined.
        """
        done = threading.Event()
        # Response with two interim results: stable prefix + speculative suffix
        multi_resp = _make_multi_result_response(
            ("안녕하세요 저는", False),
            (" 레이입니다", False),
        )

        def fake_streaming_recognize(config, requests):
            for _ in requests:
                break
            done.set()
            return iter([multi_resp])

        mock_client = MagicMock()
        mock_client.streaming_recognize.side_effect = fake_streaming_recognize
        mock_client_cls.return_value = mock_client

        asr = _make_asr()
        asr.start()
        try:
            asr.feed_audio(b"\x00" * 960)
            assert done.wait(timeout=5.0)
            _wait_for_transcript(asr, "안녕하세요 저는 레이입니다")
            assert asr.get_text() == "안녕하세요 저는 레이입니다"
        finally:
            asr.stop()

    def test_multi_result_final_plus_interim(self, mock_client_cls: MagicMock) -> None:
        """Response with a final followed by an interim in the same message."""
        done = threading.Event()
        multi_resp = _make_multi_result_response(
            ("첫 문장. ", True),
            ("두번째", False),
        )

        def fake_streaming_recognize(config, requests):
            for _ in requests:
                break
            done.set()
            return iter([multi_resp])

        mock_client = MagicMock()
        mock_client.streaming_recognize.side_effect = fake_streaming_recognize
        mock_client_cls.return_value = mock_client

        asr = _make_asr()
        asr.start()
        try:
            asr.feed_audio(b"\x00" * 960)
            assert done.wait(timeout=5.0)
            _wait_for_transcript(asr, "첫 문장. 두번째")
            assert asr.get_text() == "첫 문장. 두번째"
            with asr._lock:
                assert asr._final_transcript == "첫 문장. "
                assert asr._interim_transcript == "두번째"
        finally:
            asr.stop()

    def test_multi_result_final_clears_preceding_interims(self, mock_client_cls: MagicMock) -> None:
        """A final in a multi-result response clears interims that preceded it."""
        done = threading.Event()
        # Two interims, then a final that covers everything
        multi_resp = _make_multi_result_response(
            ("가나", False),
            ("다라", False),
            ("가나다라마", True),
        )

        def fake_streaming_recognize(config, requests):
            for _ in requests:
                break
            done.set()
            return iter([multi_resp])

        mock_client = MagicMock()
        mock_client.streaming_recognize.side_effect = fake_streaming_recognize
        mock_client_cls.return_value = mock_client

        asr = _make_asr()
        asr.start()
        try:
            asr.feed_audio(b"\x00" * 960)
            assert done.wait(timeout=5.0)
            _wait_for_transcript(asr, "가나다라마")
            assert asr.get_text() == "가나다라마"
            with asr._lock:
                assert asr._final_transcript == "가나다라마"
                assert asr._interim_transcript == ""
        finally:
            asr.stop()

    def test_empty_alternatives_skipped(self, mock_client_cls: MagicMock) -> None:
        done = threading.Event()
        empty_resp = _make_empty_response()
        valid_resp = _make_response("테스트")

        def fake_streaming_recognize(config, requests):
            for _ in requests:
                break
            done.set()
            return iter([empty_resp, valid_resp])

        mock_client = MagicMock()
        mock_client.streaming_recognize.side_effect = fake_streaming_recognize
        mock_client_cls.return_value = mock_client

        asr = _make_asr()
        asr.start()
        try:
            asr.feed_audio(b"\x00" * 960)
            assert done.wait(timeout=5.0)
            _wait_for_transcript(asr, "테스트")
            assert asr.get_text() == "테스트"
        finally:
            asr.stop()


# ---------------------------------------------------------------------------
# Tests — reset
# ---------------------------------------------------------------------------


@patch("voice_pipeline.asr.asr.speech.SpeechClient")
class TestReset:
    def test_reset_clears_and_restarts(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.streaming_recognize.return_value = iter([])
        mock_client_cls.return_value = mock_client

        asr = _make_asr()
        asr.start()
        try:
            # Simulate having a transcript
            with asr._lock:
                asr._final_transcript = "이전 텍스트"

            asr.reset()

            assert asr.get_text() == ""
            assert asr._running.is_set()
            assert asr._reader_thread is not None
        finally:
            asr.stop()

    def test_reset_when_not_running(self, mock_client_cls: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
        asr = _make_asr()
        with caplog.at_level(logging.WARNING, logger="voice_pipeline.asr"):
            asr.reset()
        assert "not running" in caplog.text

    def test_reset_during_active_stream(self, mock_client_cls: MagicMock) -> None:
        call_count = 0

        def fake_streaming_recognize(config, requests):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First stream: consume some audio then return a response
                for _ in requests:
                    break
                return iter([_make_response("첫번째")])
            # Second stream after reset
            return iter([])

        mock_client = MagicMock()
        mock_client.streaming_recognize.side_effect = fake_streaming_recognize
        mock_client_cls.return_value = mock_client

        asr = _make_asr()
        asr.start()
        try:
            asr.feed_audio(b"\x00" * 960)
            _wait_for_transcript(asr, "첫번째")

            asr.reset()

            assert asr.get_text() == ""
            assert asr._running.is_set()
            assert call_count == 2
        finally:
            asr.stop()


# ---------------------------------------------------------------------------
# Tests — error propagation
# ---------------------------------------------------------------------------


@patch("voice_pipeline.asr.asr.speech.SpeechClient")
class TestErrorPropagation:
    def test_grpc_error_propagates_via_get_text(self, mock_client_cls: MagicMock) -> None:
        from google.api_core.exceptions import InvalidArgument

        def fake_streaming_recognize(config, requests):
            raise InvalidArgument("bad config")

        mock_client = MagicMock()
        mock_client.streaming_recognize.side_effect = fake_streaming_recognize
        mock_client_cls.return_value = mock_client

        asr = _make_asr()
        asr.start()
        try:
            # Wait for reader thread to encounter the error
            _wait_for_error(asr)
            with pytest.raises(ASRError, match="bad config"):
                asr.get_text()
        finally:
            asr.stop()

    def test_grpc_error_propagates_via_feed_audio(self, mock_client_cls: MagicMock) -> None:
        from google.api_core.exceptions import InvalidArgument

        def fake_streaming_recognize(config, requests):
            raise InvalidArgument("bad config")

        mock_client = MagicMock()
        mock_client.streaming_recognize.side_effect = fake_streaming_recognize
        mock_client_cls.return_value = mock_client

        asr = _make_asr()
        asr.start()
        try:
            _wait_for_error(asr)
            with pytest.raises(ASRError, match="bad config"):
                asr.feed_audio(b"\x00" * 960)
        finally:
            asr.stop()

    def test_error_cleared_after_raise(self, mock_client_cls: MagicMock) -> None:
        from google.api_core.exceptions import InvalidArgument

        def fake_streaming_recognize(config, requests):
            raise InvalidArgument("transient")

        mock_client = MagicMock()
        mock_client.streaming_recognize.side_effect = fake_streaming_recognize
        mock_client_cls.return_value = mock_client

        asr = _make_asr()
        asr.start()
        try:
            _wait_for_error(asr)
            with pytest.raises(ASRError):
                asr.get_text()
            # After raising, the error is cleared — next call returns ""
            assert asr.get_text() == ""
        finally:
            asr.stop()


# ---------------------------------------------------------------------------
# Tests — edge cases
# ---------------------------------------------------------------------------


@patch("voice_pipeline.asr.asr.speech.SpeechClient")
class TestEdgeCases:
    def test_stop_with_thread_join_timeout(self, mock_client_cls: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
        hang_event = threading.Event()

        def fake_streaming_recognize(config, requests):
            # Block until test is done to simulate a hanging thread
            hang_event.wait(timeout=10.0)
            return iter([])

        mock_client = MagicMock()
        mock_client.streaming_recognize.side_effect = fake_streaming_recognize
        mock_client_cls.return_value = mock_client

        asr = _make_asr()
        asr.start()

        # Patch the join timeout to be very short
        def fast_stop_stream() -> None:
            if asr._audio_queue is not None:
                with contextlib.suppress(queue.Full):
                    asr._audio_queue.put_nowait(_SENTINEL)
            if asr._reader_thread is not None:
                asr._reader_thread.join(timeout=0.1)
                if asr._reader_thread.is_alive():
                    logging.getLogger("voice_pipeline.asr").warning("Reader thread did not exit within timeout")
                asr._reader_thread = None

        asr._stop_stream = fast_stop_stream  # type: ignore[method-assign]

        with caplog.at_level(logging.WARNING, logger="voice_pipeline.asr"):
            asr.stop()

        assert "did not exit" in caplog.text
        hang_event.set()  # Unblock the thread for clean test teardown

    def test_feed_audio_queue_full_drops_frame(
        self, mock_client_cls: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        block_event = threading.Event()

        def fake_streaming_recognize(config, requests):
            # Don't consume any requests — let the queue fill up
            block_event.wait(timeout=10.0)
            return iter([])

        mock_client = MagicMock()
        mock_client.streaming_recognize.side_effect = fake_streaming_recognize
        mock_client_cls.return_value = mock_client

        asr = _make_asr()
        asr.start()
        try:
            # Fill the queue
            for _ in range(300):
                asr.feed_audio(b"\x00" * 960)

            # Next feed should drop
            with caplog.at_level(logging.WARNING, logger="voice_pipeline.asr"):
                asr.feed_audio(b"\x00" * 960)
            assert "queue full" in caplog.text.lower()
        finally:
            block_event.set()
            asr.stop()

    def test_unsupported_sample_width(self, mock_client_cls: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        monkeypatch.setattr("voice_pipeline.asr.asr.SAMPLE_WIDTH", 4)
        asr = _make_asr()

        with pytest.raises(ASRError, match="sample_width=4"):
            asr.start()

        # Client must be cleaned up after start() failure
        assert asr._client is None
        mock_client.transport.close.assert_called_once()

    def test_sample_rate_too_low(self, mock_client_cls: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        monkeypatch.setattr("voice_pipeline.asr.asr.SAMPLE_RATE", 4000)
        asr = _make_asr()

        with pytest.raises(ASRError, match="sample_rate=4000"):
            asr.start()

    def test_sample_rate_too_high(self, mock_client_cls: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        monkeypatch.setattr("voice_pipeline.asr.asr.SAMPLE_RATE", 96000)
        asr = _make_asr()

        with pytest.raises(ASRError, match="sample_rate=96000"):
            asr.start()

    def test_sample_rate_boundary_valid(self, mock_client_cls: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_client.streaming_recognize.return_value = iter([])
        mock_client_cls.return_value = mock_client

        for rate in (8000, 48000):
            monkeypatch.setattr("voice_pipeline.asr.asr.SAMPLE_RATE", rate)
            asr = _make_asr()
            asr.start()
            asr.stop()


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _wait_for_transcript(asr: GoogleCloudASR, expected: str, timeout: float = 5.0) -> None:
    """Spin until get_text() matches expected value or timeout."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with asr._lock:
            current = asr._final_transcript + asr._interim_transcript
            if current == expected:
                return
        time.sleep(0.01)


def _wait_for_error(asr: GoogleCloudASR, timeout: float = 5.0) -> None:
    """Spin until an error is stored or timeout."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with asr._lock:
            if asr._error is not None:
                return
        time.sleep(0.01)
