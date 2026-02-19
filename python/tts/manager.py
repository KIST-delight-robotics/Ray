"""TTS 스트리밍 매니저. 별도 스레드에서 OpenAI TTS API를 호출합니다."""

import time
import logging
import asyncio
import threading

from config import create_openai_client, TTS_MODEL, VOICE
from protocol import RobotMessage

logger = logging.getLogger(__name__)


class TTSManager:
    def __init__(self, main_loop, websocket):
        self.client = create_openai_client()
        self.main_loop = main_loop
        self.websocket = websocket

        self.is_playing = False
        self.playback_started_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread = None
        self._current_turn_id = 0

    def speak(self, text: str, turn_id: int):
        """TTS 스트리밍 시작 (ThinkingState 호출)"""
        self._stop_event.clear()
        self.playback_started_event.clear()
        self.is_playing = True
        self._current_turn_id = turn_id

        self._thread = threading.Thread(
            target=self._run_tts,
            args=(text, turn_id),
            name="TTS_Thread",
            daemon=True
        )
        self._thread.start()

    def stop(self):
        """TTS 즉시 중단 (SpeakingState 인터럽션 호출)"""
        if self.is_playing:
            logger.info("TTS 중단 요청")
            self._stop_event.set()
            self.is_playing = False

    def _send_to_robot(self, message: str):
        """WebSocket으로 메시지 전송 (thread-safe)"""
        if self.websocket:
            asyncio.run_coroutine_threadsafe(
                self.websocket.send(message), self.main_loop
            )

    def _is_stale(self, turn_id: int) -> bool:
        """_stop_event 경합 방지: turn_id 불일치 시 stale 스레드로 판단"""
        return self._stop_event.is_set() or self._current_turn_id != turn_id

    def _run_tts(self, text: str, turn_id: int):
        stream_started = False
        try:
            tts_start_time = time.time()

            # stale 체크: 이미 취소된 턴이면 스트림을 시작하지 않음
            if self._is_stale(turn_id):
                return

            # 스트리밍 시작 알림
            self._send_to_robot(RobotMessage.tts_stream_start(turn_id))
            stream_started = True

            # OpenAI TTS 호출 (Stream)
            with self.client.audio.speech.with_streaming_response.create(
                model=TTS_MODEL, voice=VOICE, input=text, response_format="pcm"
            ) as response:
                first_chunk = True

                for chunk in response.iter_bytes(chunk_size=4096):
                    if self._is_stale(turn_id):
                        logger.info("TTS 스트리밍 루프 탈출")
                        break

                    self._send_to_robot(RobotMessage.tts_audio_chunk(chunk, turn_id))

                    if first_chunk:
                        logger.info("TTS 첫 청크 전송 -> Playback Started")
                        self.playback_started_event.set()
                        first_chunk = False

            logger.info(f"TTS 스트리밍 완료 (소요시간: {time.time() - tts_start_time:.2f}초)")

        except Exception as e:
            logger.error(f"TTS 스트리밍 오류: {e}", exc_info=True)
        finally:
            # 스트림 종료 신호: 정상/예외 모두 C++에 알려 무한 대기 방지
            if stream_started and not self._is_stale(turn_id):
                self._send_to_robot(RobotMessage.tts_stream_end(turn_id))
            self.is_playing = False
            if not self.playback_started_event.is_set():
                self.playback_started_event.set()
