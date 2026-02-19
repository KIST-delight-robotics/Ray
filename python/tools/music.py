"""음악 검색 및 재생 도구."""

import re
import json
import logging

from config import ASSETS_DIR

logger = logging.getLogger(__name__)

SONGS_DB_PATH = ASSETS_DIR / "songs_db.json"

# 모듈 로드 시 song DB 초기화
try:
    with open(SONGS_DB_PATH, 'r') as f:
        SONG_DB = json.load(f)
except FileNotFoundError:
    logger.warning(f"songs_db.json을 찾을 수 없습니다: {SONGS_DB_PATH}")
    SONG_DB = []


def _normalize_string(input_str: str) -> str:
    return re.sub(r'\s+', '', input_str).lower()


# 정규화된 검색 인덱스 생성
_song_candidates = []
for _song in SONG_DB:
    _processed = _song.copy()
    _processed['norm_title'] = _normalize_string(_song['title'])
    _processed['norm_artist'] = _normalize_string(_song['artist'])
    _song_candidates.append(_processed)


def play_music(song_title: str = "", artist_name: str = "") -> tuple[str | None, str]:
    """
    사용자가 요청한 조건에 맞는 노래를 DB에서 검색하여 반환합니다.

    Returns:
        (file_path, message) - file_path는 찾은 경우 문자열, 못 찾은 경우 None
    """
    target_title = _normalize_string(song_title)
    target_artist = _normalize_string(artist_name)

    candidates = _song_candidates

    if song_title:
        candidates = [s for s in candidates if target_title in s['norm_title']]

    if artist_name:
        candidates = [s for s in candidates if target_artist in s['norm_artist']]

    if candidates:
        selected_song = candidates[0]
        logger.info(f"재생할 노래 찾음: '{selected_song['title']}' by {selected_song['artist']}")
        return selected_song['file_path'], f"Found and playing '{selected_song['title']}' by {selected_song['artist']}."
    else:
        logger.info("재생할 노래를 찾지 못함.")
        return None, "노래를 찾을 수 없습니다."
