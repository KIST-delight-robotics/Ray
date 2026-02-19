"""C++ 로봇 클라이언트와의 WebSocket 메시지 프로토콜 정의."""

import json
import base64


class RobotMessage:
    """C++ 로봇 클라이언트로 보내는 메시지 빌더."""

    @staticmethod
    def play_audio(file_path: str) -> str:
        return json.dumps({"type": "play_audio", "file_to_play": str(file_path)})

    @staticmethod
    def play_audio_csv(audio_name: str, turn_id: int) -> str:
        return json.dumps({"type": "play_audio_csv", "audio_name": audio_name, "turn_id": turn_id})

    @staticmethod
    def user_interruption() -> str:
        return json.dumps({"type": "user_interruption"})

    @staticmethod
    def tts_stream_start(turn_id: int) -> str:
        return json.dumps({"type": "responses_only", "turn_id": turn_id})

    @staticmethod
    def tts_audio_chunk(pcm_data: bytes, turn_id: int) -> str:
        return json.dumps({
            "type": "responses_audio_chunk",
            "turn_id": turn_id,
            "data": base64.b64encode(pcm_data).decode('utf-8'),
        })

    @staticmethod
    def tts_stream_end(turn_id: int) -> str:
        return json.dumps({"type": "responses_stream_end", "turn_id": turn_id})

    @staticmethod
    def stt_done(timestamp_ms: int) -> str:
        return json.dumps({"type": "stt_done", "stt_done_time": timestamp_ms})
