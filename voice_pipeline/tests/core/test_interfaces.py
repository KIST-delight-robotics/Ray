"""Tests for voice_pipeline.core.interfaces."""

import pytest

from voice_pipeline.core.interfaces import (
    IASR,
    ILLM,
    ITTS,
    IVAP,
    IContextBuilder,
    IConversationHistory,
    ICppBridge,
    IEmbedder,
    ILEDController,
    IMemoryStorage,
    ISimilarity,
    ISpeechGenerator,
    IStorageBackend,
    ITurnDetector,
    ITurnGPT,
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

    def test_vap_abstract(self) -> None:
        with pytest.raises(TypeError):
            IVAP()  # type: ignore[abstract]

    def test_turngpt_abstract(self) -> None:
        with pytest.raises(TypeError):
            ITurnGPT()  # type: ignore[abstract]

    def test_turn_detector_abstract(self) -> None:
        with pytest.raises(TypeError):
            ITurnDetector()  # type: ignore[abstract]

    def test_speech_generator_abstract(self) -> None:
        with pytest.raises(TypeError):
            ISpeechGenerator()  # type: ignore[abstract]

    def test_similarity_abstract(self) -> None:
        with pytest.raises(TypeError):
            ISimilarity()  # type: ignore[abstract]

    def test_embedder_abstract(self) -> None:
        with pytest.raises(TypeError):
            IEmbedder()  # type: ignore[abstract]

    def test_memory_storage_abstract(self) -> None:
        with pytest.raises(TypeError):
            IMemoryStorage()  # type: ignore[abstract]
