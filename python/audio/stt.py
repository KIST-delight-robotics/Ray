"""Google Speech-to-Text 스트리밍 클라이언트."""

import time
import queue
import asyncio
import logging
import threading

from google.cloud import speech
from google.api_core import exceptions

from protocol import RobotMessage

logger = logging.getLogger(__name__)


class GoogleSTTStreamer:
    """Google Speech-to-Text API로 오디오 스트림을 처리하는 클래스."""

    def __init__(self, stt_result_queue: asyncio.Queue, main_loop: asyncio.AbstractEventLoop, websocket, sample_rate: int, stt_audio_queue: queue.Queue, stt_stop_event: threading.Event):
        self.stt_result_queue = stt_result_queue
        self.main_loop = main_loop
        self.websocket = websocket
        self.stt_audio_queue = stt_audio_queue
        self.stt_stop_event = stt_stop_event

        self.stt_client = speech.SpeechClient()
        self.stt_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="ko-KR",
            enable_automatic_punctuation=True,
        )
        self.stt_streaming_config = speech.StreamingRecognitionConfig(
            config=self.stt_config,
            interim_results=True,
            single_utterance=False,
        )
        logger.info("Google STT 클라이언트 초기화 완료")

    def _stt_audio_generator(self):
        """STT API에 오디오를 공급하는 제너레이터."""
        while not self.stt_stop_event.is_set():
            chunk = self.stt_audio_queue.get()
            if chunk is None:
                return
            data = [chunk.tobytes()]

            while True:
                try:
                    chunk = self.stt_audio_queue.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk.tobytes())
                except queue.Empty:
                    break

            yield speech.StreamingRecognizeRequest(audio_content=b''.join(data))
        logger.info("STT 오디오 공급 중단됨.")

    def run_stt_session(self):
        """단일 STT 세션을 실행하고 최종 결과를 큐에 넣음."""
        logger.info("STT 세션 스레드 시작.")

        accumulated_transcripts = []
        current_interim_transcript = ""

        try:
            audio_gen = self._stt_audio_generator()
            responses = self.stt_client.streaming_recognize(self.stt_streaming_config, audio_gen)

            for response in responses:
                if not response.results or not response.results[0].alternatives:
                    continue

                result = response.results[0]
                transcript = result.alternatives[0].transcript.strip()

                if result.is_final:
                    accumulated_transcripts.append(transcript)
                    current_interim_transcript = ""
                    logger.info(f"STT 최종 결과 조각: '{transcript}'")
                else:
                    current_interim_transcript = transcript
                    logger.info(f"STT 중간 결과: '{transcript}'")

        except exceptions.DeadlineExceeded as e:
            logger.warning(f"STT 세션 타임아웃(DeadlineExceeded): {e}")
        except Exception as e:
            logger.error(f"STT 세션 중 오류 발생: {e}", exc_info=True)
        finally:
            final_text_parts = accumulated_transcripts.copy()
            if current_interim_transcript:
                final_text_parts.append(current_interim_transcript)
            final_text = " ".join(final_text_parts).strip()

            if final_text:
                logger.info(f"STT 최종 결과 전송: '{final_text}'")
                self.stt_result_queue.put(final_text)

                if self.websocket:
                    stt_completion_time = int(time.time() * 1000)
                    asyncio.run_coroutine_threadsafe(
                        self.websocket.send(RobotMessage.stt_done(stt_completion_time)),
                        self.main_loop
                    )
            else:
                logger.info("STT 결과 없음 (빈 텍스트) -> 실패 신호 전송")
                self.stt_result_queue.put(None)

            logger.info("STT 세션 스레드 종료.")
