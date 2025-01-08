import openai
import sys
from google.cloud import texttospeech

def get_gpt_response(prompt):
    prompt_with_limit = prompt + " 2문장 이내로 답해줘."
    response = openai.ChatCompletion.create(
        model="gpt-4",  # 적절한 GPT 모델명
        messages=[
            {"role": "user", "content": prompt_with_limit}
        ],
        max_tokens=200
    )
    return response.choices[0].message["content"].strip()

def synthesize_text_to_speech(text, output_file):
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",  
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open(output_file, "wb") as out:
        out.write(response.audio_content)
        print(f"Audio content saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python gpt_tts_script.py <prompt> <output_file>")
        sys.exit(1)

    # API 키 설정
    openai.api_key = "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" 

    prompt = sys.argv[1]
    output_file = sys.argv[2]

    # 1) GPT 호출
    gpt_result = get_gpt_response(prompt)

    print(gpt_result)
    
    # 2) TTS 생성
    synthesize_text_to_speech(gpt_result, output_file)

    # 3) C++ 쪽에서 pipe로 받은 결과 로그에 찍기 위해
    #    최종 GPT 응답(텍스트)를 print 해 준다
    
