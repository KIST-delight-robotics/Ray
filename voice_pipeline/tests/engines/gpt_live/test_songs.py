"""Tests for voice_pipeline.engines.gpt_live.songs — 카탈로그 로드·파일 세트 검증·목록 포맷."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from voice_pipeline.engines.gpt_live.songs import Song, format_song_list, load_song_catalog


def _make_assets(root: Path, keys: list[str], *, drop: dict[str, str] | None = None) -> Path:
    """키마다 네 파일을 만든다. ``drop[key]`` 에 적힌 하위 폴더의 파일은 빠뜨린다."""
    files = {"audio/music": ".wav", "headMotion": ".csv", "mouthMotion": "-delta-big.csv", "ledMotion": "-led.csv"}
    for key in keys:
        for sub, suffix in files.items():
            if drop and drop.get(key) == sub:
                continue
            path = root / sub / f"{key}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
    return root


def _write_catalog(path: Path, entries: dict) -> Path:
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    return path


class TestLoadSongCatalog:
    def test_complete_entries_are_loaded_with_fields(self, tmp_path: Path) -> None:
        assets = _make_assets(tmp_path / "assets", ["IAM", "Butter_BTS"])
        catalog = _write_catalog(
            tmp_path / "songs.json",
            {
                "IAM": {
                    "title": "I AM",
                    "artist": "IVE",
                    "aliases": ["아이엠"],
                    "artist_aliases": ["아이브"],
                    "note": "x",
                },
                "Butter_BTS": {"title": "Butter", "artist": "BTS"},
            },
        )
        songs = load_song_catalog(catalog, assets)
        assert set(songs) == {"IAM", "Butter_BTS"}
        assert songs["IAM"] == Song("IAM", "I AM", "IVE", ("아이엠",), ("아이브",))
        assert songs["Butter_BTS"].aliases == ()

    def test_entry_with_missing_file_is_skipped_with_warning(self, tmp_path: Path, caplog) -> None:
        assets = _make_assets(tmp_path / "assets", ["IAM", "Nope"], drop={"Nope": "ledMotion"})
        catalog = _write_catalog(tmp_path / "songs.json", {"IAM": {"title": "I AM"}, "Nope": {"title": "No"}})
        with caplog.at_level(logging.WARNING, logger="voice_pipeline.songs"):
            songs = load_song_catalog(catalog, assets)
        assert set(songs) == {"IAM"}
        assert "Nope" in caplog.text and "ledMotion/Nope-led.csv" in caplog.text

    def test_orphan_motion_csv_is_warned(self, tmp_path: Path, caplog) -> None:
        assets = _make_assets(tmp_path / "assets", ["IAM", "Orphan"])
        catalog = _write_catalog(tmp_path / "songs.json", {"IAM": {"title": "I AM"}})
        with caplog.at_level(logging.WARNING, logger="voice_pipeline.songs"):
            load_song_catalog(catalog, assets)
        assert "Orphan" in caplog.text

    def test_missing_catalog_file_disables_songs(self, tmp_path: Path) -> None:
        assert load_song_catalog(tmp_path / "none.json", tmp_path) == {}

    def test_broken_json_or_entry_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "songs.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(RuntimeError, match="Song catalog load failed"):
            load_song_catalog(bad, tmp_path)
        _write_catalog(bad, {"IAM": {"artist": "IVE"}})  # title 없음
        with pytest.raises(RuntimeError, match="needs a title"):
            load_song_catalog(bad, tmp_path)


class TestFormat:
    def test_describe_and_list_are_one_line_per_song_sorted(self) -> None:
        songs = {
            "b": Song("b", "서랍", "10CM", ("Drawer",), ("십센치",)),
            "a": Song("a", "Nessun dorma", "", ("네순 도르마",)),
        }
        text = format_song_list(songs)
        assert text.splitlines() == ["a: Nessun dorma (네순 도르마)", "b: 서랍 — 10CM (Drawer, 십센치)"]
