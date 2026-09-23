"""노래 카탈로그 — ``assets/songs.json`` 을 읽어 재생 가능한 곡 목록을 만든다.

키 = 음원·모션 CSV 파일 stem. 네 파일 세트가 다 있는 곡만 남기고, 모르는 필드(``note`` 등)는 무시한다.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from voice_pipeline.settings import ASSETS_DIR, SONG_CATALOG_PATH

logger = logging.getLogger("voice_pipeline.songs")

# 키 하나가 가리키는 파일 세트: (assets 하위 폴더, 접미사). C++ csv_control_motor 의 경로 규칙과 같다.
_SONG_FILES: tuple[tuple[str, str], ...] = (
    ("audio/music", ".wav"),
    ("headMotion", ".csv"),
    ("mouthMotion", "-delta-big.csv"),
    ("ledMotion", "-led.csv"),
)


@dataclass(frozen=True)
class Song:
    """카탈로그 항목 하나. ``key`` 가 C++ 에 넘기는 파일 stem."""

    key: str
    title: str
    artist: str
    aliases: tuple[str, ...] = ()
    artist_aliases: tuple[str, ...] = field(default=())

    def describe(self) -> str:
        """LLM 이 읽는 한 줄: ``key: title — artist (별칭…)``."""
        names = ", ".join(a for a in (*self.aliases, *self.artist_aliases) if a)
        line = f"{self.key}: {self.title} — {self.artist}" if self.artist else f"{self.key}: {self.title}"
        return f"{line} ({names})" if names else line


def load_song_catalog(
    catalog_path: str | Path = SONG_CATALOG_PATH, assets_dir: str | Path = ASSETS_DIR
) -> dict[str, Song]:
    """카탈로그를 읽고 파일 세트가 완전한 곡만 돌려준다.

    파일이 없으면 빈 dict. 세트가 빠진 곡과 카탈로그에 없는 모션 CSV 는 경고.

    Args:
        catalog_path: ``songs.json`` 경로.
        assets_dir: 파일 세트를 찾을 assets 루트.

    Raises:
        RuntimeError: JSON 파싱 실패 또는 항목 형식 오류.
    """
    catalog_path = Path(catalog_path)
    assets = Path(assets_dir)
    if not catalog_path.exists():
        logger.info("Song catalog not found (%s) — song tools disabled", catalog_path)
        return {}
    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Song catalog load failed ({catalog_path}): {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError(f"Song catalog load failed ({catalog_path}): top level must be an object")

    songs: dict[str, Song] = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict) or not str(entry.get("title", "")).strip():
            raise RuntimeError(f"Song catalog load failed ({catalog_path}): entry {key!r} needs a title")
        missing = [
            f"{sub}/{key}{suffix}" for sub, suffix in _SONG_FILES if not (assets / sub / f"{key}{suffix}").exists()
        ]
        if missing:
            logger.warning("Song %r skipped — missing files: %s", key, ", ".join(missing))
            continue
        songs[key] = Song(
            key=key,
            title=str(entry["title"]).strip(),
            artist=str(entry.get("artist", "")).strip(),
            aliases=tuple(str(a).strip() for a in entry.get("aliases", []) if str(a).strip()),
            artist_aliases=tuple(str(a).strip() for a in entry.get("artist_aliases", []) if str(a).strip()),
        )

    orphans = sorted(p.stem for p in (assets / "headMotion").glob("*.csv") if p.stem not in raw)
    if orphans:
        logger.warning("Motion CSVs without a catalog entry (not playable): %s", ", ".join(orphans))
    logger.info("Song catalog: %d playable song(s)", len(songs))
    return songs


def format_song_list(catalog: Mapping[str, Song]) -> str:
    """백엔드 지시문·툴 설명에 넣는 목록 — 키 순서대로 한 줄씩."""
    return "\n".join(catalog[k].describe() for k in sorted(catalog))
