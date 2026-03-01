"""Tests for voice_pipeline.core.interfaces."""

import pytest

from voice_pipeline.core.interfaces import (
    IASR,
    ILLM,
    ITTS,
    IContextBuilder,
    IConversationHistory,
    ICppBridge,
    ILEDController,
    IStorageBackend,
    IUtteranceTruncator,
    IWakewordDetector,
)


class TestInterfacesAreAbstract:
    """Verify that all interfaces cannot be instantiated directly."""

    def test_storage_backend_abstract(self) -> None:
        with pytest.raises(TypeError):
            IStorageBackend()  # type: ignore[abstract]

    def test_conversation_history_abstract(self) -> None:
        with pytest.raises(TypeError):
            IConversationHistory()  # type: ignore[abstract]

    def test_utterance_truncator_abstract(self) -> None:
        with pytest.raises(TypeError):
            IUtteranceTruncator()  # type: ignore[abstract]

    def test_context_builder_abstract(self) -> None:
        with pytest.raises(TypeError):
            IContextBuilder()  # type: ignore[abstract]

    def test_asr_abstract(self) -> None:
        with pytest.raises(TypeError):
            IASR()  # type: ignore[abstract]

    def test_llm_abstract(self) -> None:
        with pytest.raises(TypeError):
            ILLM()  # type: ignore[abstract]

    def test_tts_abstract(self) -> None:
        with pytest.raises(TypeError):
            ITTS()  # type: ignore[abstract]

    def test_cpp_bridge_abstract(self) -> None:
        with pytest.raises(TypeError):
            ICppBridge()  # type: ignore[abstract]

    def test_wakeword_detector_abstract(self) -> None:
        with pytest.raises(TypeError):
            IWakewordDetector()  # type: ignore[abstract]

    def test_led_controller_abstract(self) -> None:
        with pytest.raises(TypeError):
            ILEDController()  # type: ignore[abstract]
