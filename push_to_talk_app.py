import asyncio
import base64
import sounddevice as sd

from openai import AsyncOpenAI
from openai.types.beta.realtime.session import Session
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection

from audio_util import AudioPlayerAsync, CHANNELS, SAMPLE_RATE

async def main():
    client = AsyncOpenAI()
    audio_player = AudioPlayerAsync()
    accumulated_transcripts = {}  # item_id별로 누적할 partial transcript

    async with client.beta.realtime.connect(model="gpt-4o-realtime-preview-2024-10-01") as conn:
        print("OpenAI Realtime API 연결 완료.")
        # VAD 설정 (기본)
        await conn.session.update(session={"turn_detection": {"type": "server_vad"}})

        async def read_events():
            async for event in conn:
                # 세션 생성
                if event.type == "session.created":
                    print(f"세션 생성 완료: {event.session.id}")

                # 모델의 오디오 응답 (부분)
                elif event.type == "response.audio.delta":
                    # 오디오 재생
                    bytes_data = base64.b64decode(event.delta)
                    audio_player.add_data(bytes_data)

                # 모델의 텍스트 응답 (부분)
                elif event.type == "response.audio_transcript.delta":
                    # 여기서는 부분 자막만 누적 (즉시 print X)
                    accumulated_transcripts[event.item_id] = (
                        accumulated_transcripts.get(event.item_id, "") + event.delta
                    )

                # 최종 자막이 완성되었다고 알려주는 이벤트 (가정)
                elif event.type == "response.audio_transcript.done":
                    # 완성된 자막만 출력
                    final_text = accumulated_transcripts.get(event.item_id, "")
                    print(f"[완성본] {final_text}")
                    # 혹시 다음 아이템에서 재사용하지 않도록 정리
                    accumulated_transcripts.pop(event.item_id, None)

        # 비동기로 이벤트 수신
        asyncio.create_task(read_events())

        # 마이크 입력 보내기
        stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype="int16",
        )
        stream.start()

        read_size = int(SAMPLE_RATE * 0.02)
        print("마이크 전송 시작 (Ctrl + C로 종료)")

        try:
            while True:
                if stream.read_available < read_size:
                    await asyncio.sleep(0)
                    continue

                data, _ = stream.read(read_size)
                await conn.input_audio_buffer.append(
                    audio=base64.b64encode(data).decode("utf-8")
                )
        except KeyboardInterrupt:
            print("마이크 전송 종료")
        finally:
            stream.stop()
            stream.close()

if __name__ == "__main__":
    asyncio.run(main())
