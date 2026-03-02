"""C++ audio playback bridge module."""

from voice_pipeline.bridge.cpp_bridge import CppBridge
from voice_pipeline.bridge.exceptions import BridgeError

__all__ = ["CppBridge", "BridgeError"]
