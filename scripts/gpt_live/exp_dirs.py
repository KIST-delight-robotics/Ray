"""실험 결과 폴더 — ``var/experiments/<실험>/<번호>_<태그>/``.

번호는 시작 순서대로 붙여 이름순이 곧 시간순이 되게 한다. 같은 태그 폴더가 이미 있으면 그 폴더를 쓴다 — 두 조건을
다른 프로세스로 병렬 실행해 한 태그 아래 모을 때(``exp_live_conversation.py``) 같은 폴더로 모이게.
"""

from __future__ import annotations

import re
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[2] / "var" / "experiments"
_NUMBERED = re.compile(r"^(\d+)_(.+)$")


def find_result_dir(experiment: str, tag: str) -> Path | None:
    """``<번호>_<tag>`` 폴더. 없으면 None."""
    root = EXPERIMENTS / experiment
    if not root.is_dir():
        return None
    for path in sorted(root.iterdir()):
        m = _NUMBERED.match(path.name)
        if m and m.group(2) == tag:
            return path
    return None


def result_dir(experiment: str, tag: str) -> Path:
    """태그의 결과 폴더를 돌려준다. 없으면 다음 번호로 만든다."""
    found = find_result_dir(experiment, tag)
    if found is not None:
        return found
    root = EXPERIMENTS / experiment
    root.mkdir(parents=True, exist_ok=True)
    numbers = [int(m.group(1)) for p in root.iterdir() if (m := _NUMBERED.match(p.name))]
    path = root / f"{max(numbers, default=0) + 1:02d}_{tag}"
    path.mkdir(exist_ok=True)
    return path
