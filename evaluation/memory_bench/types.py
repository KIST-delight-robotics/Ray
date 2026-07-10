"""Dataset-neutral benchmark types.

데이터셋(LoCoMo, LongMemEval, …)별 로더는 ``datasets/``에 있고, 전부 이
중립 포맷으로 변환된다. ingest/answer/score 파이프라인은 이 타입만 안다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


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
    """One benchmark question.

    ``category``는 데이터셋의 유형명 문자열 (LoCoMo: ``multi-hop`` 등,
    LongMemEval: ``question_type`` 그대로).
    """

    qa_index: int
    question: str
    answer: str
    category: str
    evidence: list[str]
    evidence_sessions: list[int]
    adversarial: bool = False
    adversarial_answer: str = ""
    question_date: str = ""  # 질문 시점 timestamp (있으면 now_fn 기준으로 사용)


@dataclass
class Conversation:
    """A full multi-session conversation with its QA set.

    ``perspectives``: 인제스트 관점 — 각 항목을 "user"로 매핑해 별도 DB를 만든다
    (LoCoMo는 두 화자 각각, LongMemEval은 ``["user"]`` 하나).
    """

    sample_id: str
    speaker_a: str
    speaker_b: str
    participants_desc: str
    perspectives: list[str]
    sessions: list[Session] = field(default_factory=list)
    qa: list[QAItem] = field(default_factory=list)
