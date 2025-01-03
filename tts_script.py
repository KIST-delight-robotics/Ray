import sys
from google.cloud import texttospeech

def synthesize_text_to_speech(text, output_file):
    client = texttospeech.TextToSpeechClient()

    # 입력된 텍스트로 TTS 요청 생성
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",  # 한국어 설정 (필요에 따라 다른 언어 코드 사용 가능)
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16  # WAV 형식
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # 출력 파일로 저장
    with open(output_file, "wb") as out:
        out.write(response.audio_content)
        print(f"Audio content saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tts_script.py <text> <output_file>")
        sys.exit(1)

    text = sys.argv[1]
    output_file = sys.argv[2]
    synthesize_text_to_speech(text, output_file)
