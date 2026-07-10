"""LoCoMo dataset loader.

``locomo10.json``을 벤치 내부의 중립 포맷(:class:`Conversation` /
:class:`Session` / :class:`Turn` / :class:`QAItem`)으로 정규화한다.

원본 포맷 주의점 (실제 파일에서 확인된 것):

- ``qa[].category``와 ``qa[].evidence``는 int/list일 때도 있고 문자열 표현
  (``"5"``, ``"['D2:8']"``)일 때도 있다 → 정규화 필수.
- category 5(adversarial)는 대부분 ``answer``가 없고 ``adversarial_answer``
  (함정 오답)만 있다.
- 세션 키는 ``session_1`` … ``session_N`` — 사전순 정렬 시 ``session_10``이
  ``session_2`` 앞에 오므로 숫자 정렬해야 한다.
- 이미지 공유 턴은 ``blip_caption``을 텍스트로 병합한다 (텍스트 전용 평가 관례).
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

CATEGORY_NAMES: dict[int, str] = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

ADVERSARIAL_CATEGORY = 5

# "1:56 pm on 8 May, 2023" — 쉼표 유무 변형까지 허용.
_DATE_FORMATS = (
    "%I:%M %p on %d %B, %Y",
    "%I:%M %p on %d %B %Y",
)

_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")
_DIA_ID_RE = re.compile(r"^D(\d+):\d+$")


@dataclass
class Turn:
    """A single utterance in a session."""

    dia_id: str
    speaker: str
    text: str


@dataclass
class Session:
    """One dated session of a conversation."""

    index: int
    date_time: str
    dt: datetime
    turns: list[Turn]

    @property
    def timestamp(self) -> str:
        """Storage-format timestamp string (``%Y-%m-%d %H:%M:%S``)."""
        return self.dt.strftime(TIMESTAMP_FORMAT)


@dataclass
class QAItem:
    """One benchmark question."""

    qa_index: int
    question: str
    answer: str
    category: int
    evidence: list[str]
    adversarial_answer: str = ""

    @property
    def is_adversarial(self) -> bool:
        return self.category == ADVERSARIAL_CATEGORY

    @property
    def evidence_session_indices(self) -> list[int]:
        """Session indices referenced by evidence dia IDs (e.g. ``D3:12`` → 3)."""
        indices: list[int] = []
        for dia_id in self.evidence:
            match = _DIA_ID_RE.match(dia_id.strip())
            if match:
                idx = int(match.group(1))
                if idx not in indices:
                    indices.append(idx)
        return indices


@dataclass
class Conversation:
    """A full multi-session conversation with its QA set."""

    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[Session] = field(default_factory=list)
    qa: list[QAItem] = field(default_factory=list)

    @property
    def speakers(self) -> tuple[str, str]:
        return (self.speaker_a, self.speaker_b)


def load_locomo(path: str | Path, sample_ids: list[str] | None = None) -> list[Conversation]:
    """Load and normalize the LoCoMo dataset.

    Args:
        path: Path to ``locomo10.json``.
        sample_ids: 지정 시 해당 ``sample_id``의 대화만 반환.

    Returns:
        Conversations in file order, sessions sorted by numeric index.
    """
    data = json.loads(Path(path).read_text())
    conversations: list[Conversation] = []
    for item in data:
        sample_id = str(item["sample_id"])
        if sample_ids is not None and sample_id not in sample_ids:
            continue
        conversations.append(_parse_conversation(item, sample_id))
    return conversations


def _parse_conversation(item: dict, sample_id: str) -> Conversation:
    raw_conv = item["conversation"]
    conv = Conversation(
        sample_id=sample_id,
        speaker_a=str(raw_conv["speaker_a"]),
        speaker_b=str(raw_conv["speaker_b"]),
    )

    indices = sorted(
        int(m.group(1)) for key in raw_conv if (m := _SESSION_KEY_RE.match(key)) and raw_conv[key] is not None
    )
    for idx in indices:
        date_time = str(raw_conv.get(f"session_{idx}_date_time", ""))
        turns = [t for t in (_parse_turn(raw) for raw in raw_conv[f"session_{idx}"]) if t.text]
        conv.sessions.append(Session(index=idx, date_time=date_time, dt=parse_session_datetime(date_time), turns=turns))

    for qa_index, raw_qa in enumerate(item.get("qa", [])):
        conv.qa.append(_parse_qa(raw_qa, qa_index))
    return conv


def _parse_turn(raw: dict) -> Turn:
    text = str(raw.get("text", "") or "").strip()
    caption = str(raw.get("blip_caption", "") or "").strip()
    if caption:
        text = f"{text} [shares a photo: {caption}]".strip()
    return Turn(dia_id=str(raw.get("dia_id", "")), speaker=str(raw.get("speaker", "")), text=text)


def _parse_qa(raw: dict, qa_index: int) -> QAItem:
    return QAItem(
        qa_index=qa_index,
        question=str(raw["question"]).strip(),
        answer=_normalize_answer_field(raw.get("answer")),
        category=int(str(raw["category"])),
        evidence=_normalize_evidence(raw.get("evidence")),
        adversarial_answer=_normalize_answer_field(raw.get("adversarial_answer")),
    )


def _normalize_answer_field(value: object) -> str:
    """Coerce answer values (str/int/list/None) to a plain string."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return str(value).strip()


def _normalize_evidence(value: object) -> list[str]:
    """Normalize evidence to a list of dia-ID strings.

    원본에는 실제 리스트와 리스트의 문자열 표현(``"['D2:8']"``)이 섞여 있다.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except (ValueError, SyntaxError):
            pass
    return [text] if text else []


def parse_session_datetime(date_time: str) -> datetime:
    """Parse a LoCoMo session date string (e.g. ``"1:56 pm on 8 May, 2023"``).

    Raises:
        ValueError: 알려진 포맷과 일치하지 않는 경우.
    """
    cleaned = " ".join(date_time.strip().split())
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized session date_time: {date_time!r}")
