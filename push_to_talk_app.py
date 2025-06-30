import asyncio
import base64
import sounddevice as sd
import wave
import queue
import sys
import json
import os
from google.cloud import speech
from openai import AsyncOpenAI
from audio_util import AudioPlayerAsync, CHANNELS, SAMPLE_RATE
import functools
import time
print = functools.partial(print, flush=True)
# Load command configuration
with open("ray_conversation.json", encoding="utf-8") as f:
    cfg = json.load(f)["Kor"]


VOICE = "ash"

# Google STT 인증 정보 설정
credential_path = "path/to/your/google-credentials.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path

# 실시간 마이크 입력 → Google STT로 streaming
async def stream_stt_to_gpt(timeout_sec: float = 5.0):
    """
    timeout_sec 초 안에 최종 STT 결과(final_text)를 못 얻으면
    빈 문자열을 리턴하고 즉시 종료합니다.
    """
    q_audio = queue.Queue()
    recording_done = asyncio.Event()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[오디오 상태] {status}", file=sys.stderr, flush=True)
        q_audio.put(bytes(indata))
        
    # ✅ 마이크 장치 자동 탐색
    input_device_index = None
    for idx, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0:
            input_device_index = idx
            break

    if input_device_index is None:
        print("[오류] 입력 가능한 마이크 장치를 찾을 수 없습니다.", file=sys.stderr)
        return ""

    device_info = sd.query_devices(input_device_index, 'input')
    SAMPLE_RATE = int(device_info['default_samplerate'])

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="ko-KR",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=True,
    )

    def audio_generator():
        while not recording_done.is_set():
            chunk = q_audio.get()
            if chunk is None:
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)
        # 스트림을 끝내기 위한 빈 청크
        yield speech.StreamingRecognizeRequest(audio_content=b"")

    start_time = time.time()
    final_text = ""

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        callback=callback
    ):
        print("🎙️ 말하세요", file=sys.stderr, flush=True)
        responses = client.streaming_recognize(streaming_config, audio_generator())

        for response in responses:
                               
            for result in response.results:
                transcript = result.alternatives[0].transcript
                start_time = time.time()
                if result.is_final:
                    final_text += transcript
                    print(f"[✅ 최종 인식 결과] {final_text}", file=sys.stderr, flush=True)
                    recording_done.set()
                    return final_text
                else:
                    print(f"[📝 중간 인식] {transcript}", file=sys.stderr, flush=True)
            # 타이머 체크
            if time.time() - start_time > timeout_sec:
                print(f"[STT] 타임아웃({timeout_sec}s) 발생, 재시작합니다.", file=sys.stderr, flush=True)
                break

    # 여기까지 오면 (timeout 이거나 스트림 종료)
    return final_text  # 빈 문자열이면 누군가 다시 호출하면 되고, 그렇지 않으면 부분 결과

# 사용자 명령 키워드 분기
def match_command(text: str):
    for cmd, keywords in cfg["conditions"].items():
        if all(kw in text for kw in keywords):
            return cmd
    return None

# 커스텀 시퀀스 실행 함수들
def exit_sequence():
    print(f"[커맨드 응답] {cfg['responses']['QUIT_PROGRAM']}")

def sing_sequence():
    print(f"[커맨드 응답] {cfg['responses']['SING_A_SONG']}")

def introduce_sequence():
    print(f"[커맨드 응답] {cfg['responses']['INTRODUCE']}")

def kidding_sequence():
    print(f"[커맨드 응답] {cfg['responses']['Kidding']}")

def weather_sequence():
    print(f"[커맨드 응답] {cfg['responses']['weather']}")

SEQUENCE_FUNCS = {
    "QUIT_PROGRAM": exit_sequence,
    "SING_A_SONG": sing_sequence,
    "INTRODUCE": introduce_sequence,
    "Kidding": kidding_sequence,
    "weather": weather_sequence,
}

# GPT Realtime API로 텍스트 전송 후 음성 응답 받기
async def send_to_gpt(text):
    client = AsyncOpenAI(api_key = "openai_api_key") # 여기에 OpenAI API 키 입력
    async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as conn:
        await conn.session.update(session={
            "instructions": "너는 애니매트로닉스 로봇이야. 너의 이름은 레이야. 내가 물어보는 것들에 대해 잘 대답해줘",
            "modalities": ["text", "audio"],
            #"turn_detection": {"type": "server_vad"},
            "voice": VOICE
        })
        print("[전송할 텍스트]", text)
        await conn.conversation.item.create(item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}]
        })
        await conn.response.create()

        with wave.open("output.wav", "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            accumulated_transcripts = {}  # item_id별로 누적할 partial transcript
            answer_done = asyncio.Event()
            async for event in conn:
                # 세션 생성
                if event.type == "session.created":
                    print(f"세션 생성 완료: {event.session.id}")

                # 모델의 오디오 응답 (부분)
                elif event.type == "response.audio.delta":
                    # 오디오 재생
                    bytes_data = base64.b64decode(event.delta)
                    #바로 재생안시킬려면, 주석 처리
                    #audio_player.add_data(bytes_data)
                    
                    #wav 파일 재생
                    wf.writeframes(bytes_data)

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

                    # 종료 신호 세트
                    answer_done.set()

                elif event.type == "response.audio.done":
                    print("[Info] GPT 답변(음성)이 모두 종료되었습니다.")
                    # 자막이 다 끝날 때까지 대기
                    try:
                        await asyncio.wait_for(answer_done.wait(), timeout=3.0)
                    except asyncio.TimeoutError:
                        print("[⚠️ Warning] 자막 응답이 2초 안에 오지 않았습니다. 그냥 종료합니다.")
                    
                    
                    

                    # WAV 파일 닫기
                    wf.close()
                    
                    await conn.close()  # 명시적으로 연결 닫기
                    #break    

                    

def print_and_exit(msg, code=0):
    # 메시지 출력 후 종료
    print(msg)
    sys.exit(code)

async def main():
    if len(sys.argv) < 2:
        print_and_exit("사용법: python push_to_talk_app.py [sleep|active]", 1)
    mode = sys.argv[1]
    if mode == "sleep":
        # Sleep 모드: "레이" 키워드 대기
        while True:
            # STT 코루틴을 별도 태스크로 실행
            task = asyncio.create_task(stream_stt_to_gpt())
            try:
                user_text = await asyncio.wait_for(task, timeout= 5)
            except asyncio.TimeoutError:
                # 일정 시간 음성 입력 없으면 계속 sleep
                print("Time out again recording", file= sys.stderr, flush=True)
                
                task.cancel()
                
                try:
                    await task
                except asyncio.CancelledError:
                    pass

                continue

            if "레이" in user_text:
                print("awake", flush=True)
                sys.exit(0)

            

    elif mode == "active":
        # Active 모드: 최대 3회, 각 5초간 STT 대기
        user_text = ""
        for attempt in range(3):
            try:
                user_text = await asyncio.wait_for(stream_stt_to_gpt(timeout_sec=5.0), timeout=5.5)
                # timeout_sec와 wait_for 타임아웃은 거의 같게 설정하세요
            except asyncio.TimeoutError:
                print(f"[STT] {attempt+1}번째 타임아웃(5s)", file=sys.stderr, flush=True)
                continue
            if user_text :
                break  # 일단 한 번이라도 음성이 들어오면 루프 종료

        # 3회 모두 타임아웃 했으면 sleep 복귀
        if not user_text:
            print("sleep", flush=True)
            sys.exit(0)

        # 들어온 텍스트가 종료/노래 키워드면 바로 리턴
        if any(kw in user_text for kw in cfg["conditions"].get("QUIT_PROGRAM", [])):
            print("sleep", flush=True)
            sys.exit(0)
        if any(kw in user_text for kw in cfg["conditions"].get("SING_A_SONG", [])):
            print(user_text + "\n", end="", flush=True)
            print("singing", flush=True)
            sys.exit(0)

        # 일반 대화 → GPT 처리
        await send_to_gpt(user_text)
        # GPT 대화 끝나면 다시 main() 으로 올라와서 sleep/active 재판단

    else:
        print_and_exit(f"알 수 없는 모드: {mode}", 1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("사용자에 의해 중단되었습니다.", file=sys.stderr, flush=True)
        sys.exit(0)
