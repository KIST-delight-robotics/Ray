import os
import time
import asyncio
import logging

from openai import AsyncOpenAI

from config import TTS_MODEL, VOICE

logger = logging.getLogger(__name__)


async def save_tts_to_file(response_text: str, client: AsyncOpenAI, filename: str = "output.mp3"):
    """텍스트를 받아 TTS 오디오 파일로 저장"""
    try:
        tts_start_time = time.time()
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        async with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=VOICE,
            input=response_text,
            response_format="wav"
        ) as tts_response:

            logger.info(f"TTS 파일 저장 시작: {filename}")

            with open(filename, "wb") as f:
                async for audio_chunk in tts_response.iter_bytes(chunk_size=4096):
                    if audio_chunk:
                        f.write(audio_chunk)

        logger.info(f"TTS 파일 '{filename}' 저장 완료 (소요시간: {time.time() - tts_start_time:.2f}초)")

    except asyncio.CancelledError:
        logger.info("TTS 처리가 중단되었습니다.")
        if os.path.exists(filename):
            os.remove(filename)
        raise
    except Exception as e:
        logger.error(f"TTS 저장 중 오류 발생: {e}")
