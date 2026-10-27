"""GPT-Live 세션 지시문 — 대화 모델 instructions, 백엔드 instructions, 라디오 문구·판정기, 목소리·인사 문구.

세션 시작 후에는 바꿀 수 없으므로 프로필·최근 세션 블록(:mod:`voice_pipeline.session_context`)을
:func:`build_live_instructions` 로 시작 지시문 뒤에 한 번 붙인다. 노래 재생 중 지시(:data:`SONG_PLAYING_INSTRUCTIONS`)
는 그때 ``instructions.append`` 로 얹고 끝나면 종료 지시로 전환한다. 백엔드 지시문은
:func:`build_backend_instructions` 가 만든다.

라디오(:mod:`~voice_pipeline.engines.gpt_live.radio`): 대화 모델은 라디오 진행 규칙을 모른다 — 라이브 지시문에는
위임 목록과 Backend tools 의 Radio 한 줄씩만 있고, 앱이 대사를 :data:`RADIO_LINE_TEMPLATE` 로 하나씩 앞서
``thinking.append`` 하면 그대로 읽는다. 멈춤·재개·종료는 상태 문구(:data:`RADIO_PAUSED_NOTE` 등)를 thinking 으로
알린다. 사용자가 말하기 시작한 뒤 대사가 들어갔으면 :data:`RADIO_LATE_LINE_NOTE` 로 대답을 먼저 하게 한다. 백엔드는
``## Radio talk`` 절의 상태로 ``start_radio`` / ``resume_radio`` 를 가린다. 끼어든 말의 판정은 별도
LLM(:data:`RADIO_JUDGE_INSTRUCTIONS`)이, 멈춘 대화가 잠잠해졌을 때 라디오로 돌아갈지는 또 다른
LLM(:data:`RADIO_RESUME_JUDGE_INSTRUCTIONS`)이 한다. 스스로 돌아갈 때는 첫 대사 앞에
:data:`RADIO_AUTO_RESUME_OPENING` 을 붙인다. 설계 배경은 docs/dj-mode-worknote.md.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from voice_pipeline.engines.gpt_live.songs import Song, format_song_list
from voice_pipeline.session_context import format_current_session_marker

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
- Something the user told Ray in an earlier session: search_memory, with the query in English. \
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
# 시작·재개 때 — 말투만 적는다. "대사·읽다" 같은 낭독 어휘는 말로 새어 나왔다(dj-mode-worknote.md 관찰)
RADIO_ON_NOTE = "Radio talk: on. Speak slowly and softly, in a gentle, unhurried voice."
RADIO_ENDED_NOTE = "Radio talk: ended. Talk with the user now."
# 사용자 전사가 오기 직전(목소리는 이미 시작)에 넣은 대사 — 모델은 사용자 말 도중 받은 대사를 대답보다 먼저 읽는다
RADIO_LATE_LINE_NOTE = (
    "The user started talking to you. Respond to the user first; do not say the line you were just given yet."
)
# 곡 뒤 대본의 첫 대사 앞에 붙인다 — 대본은 놓이는 순서와 상관없이 쓴다
RADIO_AFTER_SONG_OPENING = "노래 잘 들으셨어요?"

# 끼어든 말의 판정기 (별도 LLM 호출, 구조화 출력). 입력은 읽던 대사와 끼어든 뒤의 사용자·레이 전사.
# 기본값은 on(판정 실패·지연도 on) — 판정기는 사용자가 레이에게 말한 게 분명할 때만 paused 를 낸다.
RADIO_JUDGE_INSTRUCTIONS = """\
Ray, a small desk robot, is hosting a radio talk: it reads prepared lines aloud, one at a time. \
Someone spoke while Ray was reading. The talk goes on by default; decide whether it must pause because the user \
is talking to Ray. The transcripts come from speech recognition and may contain errors.

- paused: the user is clearly talking to Ray: a question (also about the talk itself), a request other than to \
go on, asking Ray to wait or to stop the talk, a complaint about the talk, or a comment or their own story in \
response to the talk (for example 나도 그런 적 있어) — even if Ray kept reading. Ray stops the talk and talks with \
the user.
- on: everything else — a brief reaction (for example 응, 아 진짜, 맞아, 그렇구나), asking Ray to go on with the \
talk (for example 계속해, 이어서 해 줘), talking on the phone or to someone else, talking to themselves, or words \
that are unclear. The talk goes on.
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


def build_radio_judge_prompt(
    line: str, exchange: Sequence[tuple[str, str]], earlier: Sequence[tuple[str, str]] = ()
) -> str:
    """판정기 사용자 메시지. ``exchange`` 는 끼어든 뒤의 (화자 user/assistant, 전사) 차례, ``earlier`` 는 같은 대사
    안에서 앞서 있었던 끼어들기의 차례(그 뒤 이어 읽은 낭독은 빼고) — 두 번째 말이 레이에게 한 대답인지 알게."""

    def dialog(turns: Sequence[tuple[str, str]]) -> str:
        return "\n".join(f"{'User' if who == 'user' else 'Ray'}: {text}" for who, text in turns)

    text = f"Line Ray was reading: {line}\n\n"
    if earlier:
        text += f"Earlier during this line:\n{dialog(earlier)}\n\n"
    return text + f"Since the user spoke:\n{dialog(exchange)}"


# 멈춘 대화가 잠잠해졌을 때 레이가 라디오로 스스로 돌아갈지 정하는 판정기 (별도 LLM 호출, 추론 low, 구조화 출력).
# 입력은 태그 셋으로 나누고 지시문은 태그 이름으로만 가리킨다 — 이름에 설명을 섞거나 예시 단어를 주면 판정기가 겉글자로
# 맞춰 봤다(dj-mode-worknote.md 관찰 "복귀 판정 벤치"). 라디오 토크를 준비된 대사로만 정의해야 같은 주제의 대화를
# 라디오로 보지 않는다.
RADIO_RESUME_JUDGE_INSTRUCTIONS = """\
Ray, a small desk robot, hosts a radio talk: it reads prepared lines aloud, one at a time. The radio talk is only \
those lines. While Ray was reading a line, the user said something to Ray, so the reading paused and they had a \
conversation. The conversation has now gone quiet. Decide whether Ray goes back to reading the radio talk on its own \
now. The transcripts come from speech recognition and may contain errors.

The input has three parts:
- <radio_line>: the line Ray was reading.
- <interruption>: what the user said while Ray was reading it. This paused the talk.
- <conversation>: everything said after that, in order. Ray's turns here are replies in the conversation, not the \
radio talk, even when they are about the same topic. Each of Ray's turns is cut to its last sentence.

Decisions:
- go_back: the default. Ray goes back to the radio talk, whatever the conversation was about, unless stay_paused \
applies.
- wait_for_answer: Ray's last turn in <conversation> asked the user something or offered something, and the user \
has not answered yet.
- stay_paused: only in one of these two cases. Ray then waits until the user asks to go on with the talk.
  1. The user asked to end the radio talk and did not take it back later. <interruption> counts, because Ray was \
reading the talk then. In <conversation>, only words that mention the radio talk or answer Ray's question about it \
count; words that stop or close Ray's reply do not.
  2. The last thing the user said to Ray asks Ray to wait until they are back or speak again, for example to get a \
word in, step away or take a call. This holds however long it has been, and ends once the user says anything more \
to Ray. Talking to someone else, such as on the phone, is not saying something to Ray.

If <interruption> could be either a request to end the talk (case 1) or a request to stop so the user can talk \
(case 2), look at <conversation>: if the user talked with Ray about what the talk was about, take it as case 2; \
otherwise take it as case 1.
"""
# wait_for_answer 뒤 다시 판정할 때 지시문 끝에 붙인다
RADIO_RESUME_JUDGE_SECOND = (
    "\nRay has already waited for an answer, so wait_for_answer is not an option. If waiting for an answer was the "
    "only reason not to go back, choose go_back."
)
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.?!])\s*(?=[^.?!\s])")  # 전사는 문장 사이 공백이 없기도 하다 ("요.길가")


def radio_resume_judge_format(*, second: bool = False) -> dict[str, Any]:
    """복귀 판정기 출력 형식. ``second`` 면 wait_for_answer 가 없다."""
    choices = ["go_back", "stay_paused"] if second else ["go_back", "wait_for_answer", "stay_paused"]
    return {
        "type": "json_schema",
        "name": "radio_resume",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "One short sentence."},
                "decision": {"type": "string", "enum": choices},
            },
            "required": ["reason", "decision"],
            "additionalProperties": False,
        },
    }


def build_radio_resume_prompt(line: str, interruption: str, conversation: Sequence[tuple[str, str]]) -> str:
    """복귀 판정기 사용자 메시지. ``conversation`` 은 멈춘 뒤의 (화자 user/assistant, 전사) 차례 — 레이 말은 끝
    문장만 넣는다: 사용자가 끊거나 답한 건 대개 끝이고, 대화가 길어도 앞의 그만 요청이 묻히지 않게."""

    def turn(who: str, text: str) -> str:
        if who == "user":
            return f"User: {text}"
        return f"Ray: {[s for s in _SENTENCE_BREAK_RE.split(text.strip()) if s][-1]}"

    rows = "\n".join(turn(who, text) for who, text in conversation if text.strip())
    return f"<radio_line>{line}</radio_line>\n<interruption>{interruption}</interruption>\n" + (
        f"<conversation>\n{rows}\n</conversation>" if rows else "<conversation></conversation>"
    )


# 자동 복귀 때 첫 대사 앞에 붙인다 — 모델에게 맡기면 방송 어휘("라디오 다시 켤게요")가 나왔다
RADIO_AUTO_RESUME_OPENING = "그럼 하던 이야기로 돌아가 볼게요."


# 종료 시퀀스에서 모델이 작별 인사 없이 조용하면 commentary 로 넣는다 (모델이 자기 말로 옮겨 말한다)
FAREWELL_COMMENTARY = "Say goodbye to the user now, in one short sentence: 다음에 또 불러주세요."

LIVE_VOICE = "cedar"  # 세션 목소리. 인사 WAV 도 같은 목소리로 합성해 이질감을 없앤다
LIVE_GREETING_TEXT = "네, 부르셨어요?"  # 웨이크워드 뒤 세션 연결 지연(1.5~3.5초)을 가리는 인사
LIVE_GREETING_TTS_MODEL = "gpt-4o-mini-tts"  # Live 목소리(cedar)를 지원하는 OpenAI TTS 모델


def build_live_instructions(profile_text: str = "", recent_session_texts: Sequence[str] = (), *, now: datetime) -> str:
    """세션 시작 instructions — 기본 지시문 뒤에 프로필(블록 2)과 최근 세션(블록 3)을 붙이고 현재 세션 표시로 끝낸다.

    instructions 는 세션 시작 후 바꿀 수 없으므로 세션 수준 컨텍스트는 여기에 한 번 들어간다.
    한도 16,384 토큰에 대해 프로필 256 + 최근 세션 512 soft cap 이라 여유가 크다. 대화 모델은 시계가 없어
    끝의 현재 세션 표시가 "지금" 의 유일한 근거다 — 백엔드도 이 지시문을 같이 받는다(FINDINGS §4).

    Args:
        profile_text: :func:`~voice_pipeline.session_context.format_profile_block` 결과. 빈 문자열이면 생략.
        recent_session_texts: 최근 세션 블록 텍스트, 시간순. 비어 있으면 생략.
        now: 세션 시작 시각(현지 시간대).
    """
    parts = [DEFAULT_LIVE_INSTRUCTIONS.rstrip()]
    if profile_text or recent_session_texts:
        parts.append("What you already know about the user, from earlier sessions:")
    if profile_text:
        parts.append(profile_text.strip())
    parts.extend(text.strip() for text in recent_session_texts if text.strip())
    parts.append(format_current_session_marker(now))
    return "\n\n".join(parts) + "\n"
