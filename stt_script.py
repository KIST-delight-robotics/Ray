import sys
from google.cloud import speech

def transcribe_audio(file_path):
    client = speech.SpeechClient()
    with open(file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="ko-KR",
    )

    response = client.recognize(config=config, audio=audio)

    # 가장 높은 확률의 대안 텍스트만 출력
    for result in response.results:
        print(result.alternatives[0].transcript) #alterantives 안에 여러 대안들이 들어있는데, 이중 제일 확률 높은 걸 출력

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stt_script.py <audio_file_path>")
        sys.exit(1)

    transcribe_audio(sys.argv[1])
