"""GPT-Live 세션 지시문 — 대화 모델 instructions, 백엔드 instructions, 라디오 문구·판정기, 목소리·인사 문구.

세션 시작 후에는 바꿀 수 없으므로 프로필·최근 세션 블록(:mod:`voice_pipeline.session_context`)을
:func:`build_live_instructions` 로 시작 지시문 뒤에 한 번 붙인다. 노래 재생 중 지시(:data:`SONG_PLAYING_INSTRUCTIONS`)
는 그때 ``instructions.append`` 로 얹고 끝나면 종료 지시로 전환한다. 백엔드 지시문은
:func:`build_backend_instructions` 가 만든다.

라디오(:mod:`~voice_pipeline.engines.gpt_live.radio`): 대화 모델은 라디오 진행 규칙을 모른다 — 라이브 지시문에는
위임 목록과 Backend tools 의 Radio 한 줄씩만 있고, 앱이 대사를 :data:`RADIO_LINE_TEMPLATE` 로 하나씩 앞서
``thinking.append`` 하면 그대로 읽는다. 멈춤·재개·종료는 상태 문구(:data:`RADIO_PAUSED_NOTE` 등)를 thinking 으로
알린다. 백엔드는 ``## Radio talk`` 절의 상태로 ``start_radio`` / ``resume_radio`` 를 가린다. 끼어든 말의 판정은 별도
LLM(:data:`RADIO_JUDGE_INSTRUCTIONS`)이 한다. 설계 배경은 docs/dj-mode-worknote.md.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from voice_pipeline.engines.gpt_live.songs import Song, format_song_list

# ---------------------------------------------------------------------------
# Prompts — 라이브 지시문은 공식 템플릿 구조(페르소나 → Backchannel / Interruption / Delegation policy)를 줄인 것.
# 위임 정책은 한 벌이고 Backend tools 목록과 위임 목록이 짝을 이룬다 — 목록에서 툴 줄을 빼면 백엔드 툴 선택이
# 흔들렸다(dj-mode-worknote.md "백엔드 툴 선택"). 라디오는 그 두 목록에 한 줄씩만 둔다. Radio 줄은 라디오가 레이 자신이
# 진행하는 것이라고 정의한다 — 없으면 시작 한마디가 "라디오 틀어볼게요"처럼 기기를 켜는 말이 됐다.
# 라이브·백엔드 지시문은 각각 1단계 제목 하나 아래에 둔다 — 백엔드는 둘을 이어 붙여 받으므로(FINDINGS §4)
# 같은 급의 블록으로 보이게.
# ---------------------------------------------------------------------------

DEFAULT_LIVE_INSTRUCTIONS = """\
# Voice model (Ray) instructions
You are Ray, a small, friendly desk robot.
Speak Korean unless the user asks to switch. Speak warmly and naturally, at an unhurried pace.
Default to polite Korean (해요체). If the user profile below states a tone preference, follow that instead.
If the user is frustrated, acknowledge it briefly and focus on the next helpful step.

Backchannel policy: Use moderate backchannels. Acknowledge naturally without competing with the main response.

Interruption policy: Stop speaking when the user interrupts. Listen to what they say.

Delegation policy:
Backend tools:
- Web search: current date and time, weather, news, and facts you are not sure about.
- Earlier sessions: things the user told you in earlier sessions.
- Device settings: your speaker volume and your light (brightness, or off).
- Music: playing or stopping one of your songs, and telling which songs you have.
- Radio: your own radio talk, which you host and speak yourself. Starting it, or resuming it after a pause.
- End of conversation: closes the session when the user is done talking.

Delegate to the backend when:
- The request needs current information or a fact you are not sure about.
- The user asks about an earlier session, and it is not in what you already know about the user.
- The user asks to change or check the volume or the light, including whether the light is on.
- The user asks to play or stop music, even without naming a song, or asks which songs you have.
- The user says goodbye or wants to end the conversation. Say a short goodbye yourself at the same time.
- The user asks to start the radio, or to continue a paused radio talk.

Do not delegate to the backend when:
- You can answer from the conversation, from what you already know about the user, or a still-current result.
- You need a brief clarification to understand the request.

Delegate before giving an answer that depends on backend work.
Do not guess the result while waiting.
"""

# 기본 백엔드 프롬프트
DEFAULT_BACKEND_INSTRUCTIONS = """\
# Backend instructions
## Voice conversation context
You are the backend for Ray, a Korean-speaking desk robot, in a live voice conversation.
Transcripts can contain mistakes, unfinished phrases, and later corrections. Use the latest context. \
If a needed detail is still unclear, ask for that detail instead of guessing.

## Task instructions
Information
- Current facts, weather, news, dates: web search.
- Something the user told Ray in an earlier session: search_memory, with the query in the user's language. \
Use only the memories that match the question. If nothing matches, say Ray does not remember; do not web-search it.

Device
- Louder or quieter: adjust_volume, steps=1 normally, steps=2 for "a lot".
- Brighter, dimmer or a named level: set_brightness with the level next to the current one. \
Call get_device_settings first only if the current level is not in the conversation, and alone when the user \
asks how loud or bright Ray is or whether the lights are on (level "off" means off).

Music (see Songs on Ray)
- User asks which songs Ray has, or wants music without naming a song: do not play. Name three or four titles \
with artists that fit the conversation, say there are more, and ask which one.
- User names a stored song, or agrees to one Ray proposed: play_song with its id. Never choose for them unless asked.
- No stored song matches: say Ray does not have it. Unclear which one: ask.

Radio (see Radio talk)
- Ray hosts its own radio talk: the application sends Ray lines and Ray reads them aloud.
- The user asks to start the radio (라디오, 디제이, DJ, 혼자 진행): start_radio. \
This starts a new talk, also while a talk is paused.
- The radio talk is paused and the user asks to go on with it \
(계속해, 이어서 해 줘, 이야기 계속해 줘, 라디오 이어서 해 줘): resume_radio.
- Do not call start_radio or resume_radio in any other case.

Ending
- User says goodbye or wants to stop talking, in their own words in the transcript: end_conversation. \
Never call it for any other reason.

## Return the result
Return the relevant facts, the task's current status, and the next step.
Report an action as complete only after the tool confirms it; never say it will be or was done without the result.
Likewise never state the current volume or brightness without a get_device_settings result.
After start_radio, resume_radio, play_song or end_conversation, return an empty message.
If a device result has at_limit or moved 0, say it is already at the maximum or minimum.
Answer in Korean, in one or two short sentences that sound natural when spoken aloud.
No markdown, bullet lists or URLs; naming a few songs inside a sentence is fine.
"""

# 노래 재생 중 백엔드 프롬프트
SONG_BACKEND_INSTRUCTIONS = """\
## Role
You are a classifier, not a conversation partner. A song is playing on Ray's speaker.
Your only job is to decide whether the user is asking to stop the song.

## Input
You receive the voice conversation transcript. It may contain mistakes, and the microphone picks up the music \
and nearby voices. Only the user's latest words matter.

## Decision
- If the user wants the song stopped: call stop_song.
- Otherwise: do nothing.

## Output
After calling stop_song, reply: "The song has been stopped." Otherwise reply with an empty message.
"""

# 노래 재생 시 삽입
SONG_PLAYING_INSTRUCTIONS = (
    "A song is playing.\n"
    "Respond only when the user wants the song stopped, by delegating to the backend without speaking.\n"
    "Otherwise, keep listening and stay silent until the song ends.\n"
    "Do not treat the music or nearby conversation as a request."
)
# 노래 재생 종료 시 삽입
SONG_ENDED_INSTRUCTIONS = "The song has ended. Continue the conversation as usual."


def build_songs_section(catalog: Mapping[str, Song]) -> str:
    """``## Songs on Ray`` 절 — 백엔드가 목록 질문에 답하고 ``play_song`` id 를 고르는 단일 출처.

    카탈로그가 비면 빈 문자열.
    """
    if not catalog:
        return ""
    return "## Songs on Ray\nOne per line: song id: title — artist (aliases).\n" + format_song_list(catalog) + "\n"


RADIO_STATUSES = ("off", "on", "paused")


def build_radio_section(status: str) -> str:
    """``## Radio talk`` 절 — 백엔드가 시작·재개를 가르는 상태. 상태가 바뀔 때마다 ``session.update`` 로 바꾼다."""
    if status not in RADIO_STATUSES:
        raise ValueError(f"unknown radio status {status!r}")
    return f"## Radio talk\nRadio talk status: {status}\n"


def build_backend_instructions(catalog: Mapping[str, Song] | None = None, radio_status: str | None = None) -> str:
    """백엔드 지시문 — 기본 지시문 뒤에 곡 목록 절, (라디오를 쓸 수 있으면) 라디오 상태 절을 붙인다.

    레이 말투·사용자 말투 선호는 따로 넣지 않는다 — 백엔드는 대화 모델의 세션 지시문(프로필 블록 포함)을 함께 받는다
    (FINDINGS §4).

    Args:
        catalog: 재생 가능한 곡. 비면 곡 목록 절 생략.
        radio_status: 라디오 상태(off / on / paused). None 이면 라디오가 없다(대본 없음).
    """
    sections = build_songs_section(catalog or {})
    if radio_status is not None:
        sections += build_radio_section(radio_status)
    return DEFAULT_BACKEND_INSTRUCTIONS + "\n" + sections


# ---------------------------------------------------------------------------
# 라디오 — 대화 모델에 넣는 thinking 문구. 문구에 민감하다(dj-mode-worknote.md 관찰):
# 대사 틀에 "radio"·"next line" 을 넣으면 모델이 틀을 말로 옮겨("지금 라디오에서는…", "이어서 한마디 더 할게요")
# 지금 하는 말로만 적는다. 상태 문구는 재개를 모델 혼자 할 수 없는 일로 적는다 — 위임은 그런 일에서만 걸렸다.
# ---------------------------------------------------------------------------

RADIO_LINE_TEMPLATE = "You are now saying: «{}»"
RADIO_PAUSED_NOTE = (
    "Radio talk: paused, because the user started talking to you. Talk with the user now. "
    "Do not continue the talk yourself: the rest of it is not with you. If the user asks to go on with it "
    "(for example 계속해, 이어서 해 줘, 이야기 계속해 줘), delegate to the backend to resume it."
)
RADIO_RESUMED_NOTE = "Radio talk: resumed. Read the application's lines as before."
RADIO_ENDED_NOTE = "Radio talk: ended. Talk with the user now."
# 곡 뒤 대본의 첫 대사 앞에 붙인다 — 대본은 놓이는 순서와 상관없이 쓴다
RADIO_AFTER_SONG_OPENING = "노래 잘 들으셨어요?"

# 끼어든 말의 판정기 (별도 LLM 호출, 구조화 출력). 입력은 읽던 대사와 끼어든 뒤의 사용자·레이 전사.
RADIO_JUDGE_INSTRUCTIONS = """\
Ray, a small desk robot, is hosting a radio talk: it reads prepared lines aloud, one at a time. \
The user spoke while Ray was reading. Decide what the radio should do now, from what the user said and \
how Ray responded. The transcripts come from speech recognition and may contain errors.

- on: the user only reacted briefly (for example 응, 아 진짜, 맞아, 그렇구나), or Ray ignored it and kept reading. \
The talk goes on with the next line.
- paused: the user is talking to Ray: a question (also about the talk itself), a comment, a complaint, a request, \
asking Ray to wait, or asking to stop the talk. Ray stops the talk and talks with the user.
"""
RADIO_JUDGE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "name": "radio_state",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "One short sentence."},
            "state": {"type": "string", "enum": ["on", "paused"]},
        },
        "required": ["reason", "state"],
        "additionalProperties": False,
    },
}


def build_radio_judge_prompt(line: str, exchange: Sequence[tuple[str, str]]) -> str:
    """판정기 사용자 메시지. ``exchange`` 는 끼어든 뒤의 (화자 user/assistant, 전사) 차례."""
    dialog = "\n".join(f"{'User' if who == 'user' else 'Ray'}: {text}" for who, text in exchange)
    return f"Line Ray was reading: {line}\n\nSince the user spoke:\n{dialog}"


# 종료 시퀀스에서 모델이 작별 인사 없이 조용하면 commentary 로 넣는다 (모델이 자기 말로 옮겨 말한다)
FAREWELL_COMMENTARY = "Say goodbye to the user now, in one short sentence: 다음에 또 불러주세요."

LIVE_VOICE = "cedar"  # 세션 목소리. 인사 WAV 도 같은 목소리로 합성해 이질감을 없앤다
LIVE_GREETING_TEXT = "네, 부르셨어요?"  # 웨이크워드 뒤 세션 연결 지연(1.5~3.5초)을 가리는 인사
LIVE_GREETING_TTS_MODEL = "gpt-4o-mini-tts"  # Live 목소리(cedar)를 지원하는 OpenAI TTS 모델


def build_live_instructions(profile_text: str = "", recent_session_texts: Sequence[str] = ()) -> str:
    """세션 시작 instructions — 기본 지시문 뒤에 프로필(블록 2)과 최근 세션(블록 3)을 붙인다.

    instructions 는 세션 시작 후 바꿀 수 없으므로 세션 수준 컨텍스트는 여기에 한 번 들어간다.
    한도 16,384 토큰에 대해 프로필 256 + 최근 세션 512 soft cap 이라 여유가 크다.

    Args:
        profile_text: :func:`~voice_pipeline.session_context.format_profile_block` 결과. 빈 문자열이면 생략.
        recent_session_texts: 최근 세션 블록 텍스트, 시간순. 비어 있으면 생략.
    """
    parts = [DEFAULT_LIVE_INSTRUCTIONS.rstrip()]
    if profile_text or recent_session_texts:
        parts.append("What you already know about the user, from earlier sessions:")
    if profile_text:
        parts.append(profile_text.strip())
    parts.extend(text.strip() for text in recent_session_texts if text.strip())
    return "\n\n".join(parts) + "\n"
