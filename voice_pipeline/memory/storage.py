"""Memory storage implementations.

SQLiteMemoryStorage: production backend using the shared ray.db.
InMemoryMemoryStorage: in-memory backend for unit tests.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np

from voice_pipeline.core.interfaces import IMemoryStorage
from voice_pipeline.memory.types import Episode, Profile

logger = logging.getLogger("voice_pipeline.memory")

_DEFAULT_DIMENSION = 384  # 기본 embedding 차원 (all-MiniLM-L6-v2 기준)
_DEFAULT_DB_PATH = "data/ray.db"  # 기본 SQLite 파일 경로 (History/Trace와 공유)


class SQLiteMemoryStorage(IMemoryStorage):
    """SQLite-backed memory storage.

    Opens its own connection to the shared DB file (WAL mode allows
    concurrent connections). Manages episodes, profiles, utterances,
    and FTS5 index tables.

    Thread-safe: a lock serializes all connection access so that
    concurrent callers (orchestrator main thread, retriever background
    thread, write-executor thread) do not corrupt the connection state.
    """

    def __init__(self, db_path: str, *, dimension: int = _DEFAULT_DIMENSION) -> None:
        self._db_path = db_path
        self._dimension = dimension
        self._lock = threading.Lock()
        self._conn = self._open_db(db_path)
        self._create_tables()
        self._migrate()

    def _open_db(self, db_path: str) -> sqlite3.Connection:
        """Open DB with graduated corruption recovery.

        Mirrors SQLiteStorageBackend._open_db() pattern.
        """
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Step 1: Normal open
        try:
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("PRAGMA integrity_check")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            return conn
        except sqlite3.DatabaseError:
            logger.warning("Memory DB integrity check failed, attempting WAL recovery")

        # Step 2: Delete WAL and retry
        wal = Path(db_path + "-wal")
        shm = Path(db_path + "-shm")
        if wal.exists() or shm.exists():
            if wal.exists():
                wal.unlink()
            if shm.exists():
                shm.unlink()
            try:
                conn = sqlite3.connect(db_path, check_same_thread=False)
                conn.execute("PRAGMA integrity_check")
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                logger.warning("WAL recovery succeeded — recent writes may be lost")
                return conn
            except sqlite3.DatabaseError:
                logger.warning("WAL recovery failed, creating new DB")

        # Step 3: Backup corrupt file and create new
        if path.exists():
            backup = Path(db_path + ".corrupt")
            path.rename(backup)
            logger.error("DB corrupted — backed up to %s, creating new DB", backup)

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                text            TEXT    NOT NULL,
                timestamp       TEXT    NOT NULL,
                session_id      TEXT    NOT NULL,
                importance      REAL    NOT NULL DEFAULT 0.5,
                last_cited_at   TEXT    NOT NULL,
                citation_count  INTEGER NOT NULL DEFAULT 0,
                embedding       BLOB
            );

            CREATE INDEX IF NOT EXISTS idx_episodes_session
                ON episodes(session_id);

            CREATE TABLE IF NOT EXISTS profiles (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                topic      TEXT NOT NULL,
                sub_topic  TEXT NOT NULL,
                content    TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS utterances (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                role        TEXT    NOT NULL,
                text        TEXT    NOT NULL,
                timestamp   TEXT    NOT NULL,
                token_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_utterances_session
                ON utterances(session_id);

            CREATE TABLE IF NOT EXISTS processed_sessions (
                session_id  TEXT PRIMARY KEY,
                processed_at TEXT NOT NULL
            );
        """)
        # FTS5 virtual table — must be created outside executescript
        # because CREATE VIRTUAL TABLE cannot be in a multi-statement script
        # on some SQLite builds.
        try:
            self._conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
                    USING fts5(text, content='episodes', content_rowid='id')
            """)
        except sqlite3.OperationalError:
            logger.debug("episodes_fts already exists or FTS5 unavailable", exc_info=True)

        # Triggers to keep FTS in sync with episodes table
        for trigger_sql in (
            """CREATE TRIGGER IF NOT EXISTS episodes_ai
               AFTER INSERT ON episodes BEGIN
                   INSERT INTO episodes_fts(rowid, text) VALUES (new.id, new.text);
               END""",
            """CREATE TRIGGER IF NOT EXISTS episodes_ad
               AFTER DELETE ON episodes BEGIN
                   INSERT INTO episodes_fts(episodes_fts, rowid, text)
                       VALUES ('delete', old.id, old.text);
               END""",
            """CREATE TRIGGER IF NOT EXISTS episodes_au
               AFTER UPDATE OF text ON episodes BEGIN
                   INSERT INTO episodes_fts(episodes_fts, rowid, text)
                       VALUES ('delete', old.id, old.text);
                   INSERT INTO episodes_fts(rowid, text) VALUES (new.id, new.text);
               END""",
        ):
            with contextlib.suppress(sqlite3.OperationalError):
                self._conn.execute(trigger_sql)
        self._conn.commit()

    def _migrate(self) -> None:
        """Apply schema migrations for columns added after initial release."""
        migrations = [
            "ALTER TABLE episodes ADD COLUMN citation_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE utterances ADD COLUMN token_count INTEGER NOT NULL DEFAULT 0",
        ]
        for sql in migrations:
            try:
                self._conn.execute(sql)
                self._conn.commit()
            except sqlite3.OperationalError:
                # Column already exists — expected after first migration.
                pass

    # --- Episode ---

    def add_episode(self, episode: Episode) -> int | None:
        """Persist a new episode."""
        embedding_blob = episode.embedding.astype(np.float32).tobytes() if episode.embedding is not None else None
        with self._lock:
            try:
                cursor = self._conn.execute(
                    "INSERT INTO episodes "
                    "(text, timestamp, session_id, importance, last_cited_at, "
                    "citation_count, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        episode.text,
                        episode.timestamp,
                        episode.session_id,
                        episode.importance,
                        episode.last_cited_at,
                        episode.citation_count,
                        embedding_blob,
                    ),
                )
                self._conn.commit()
                return cursor.lastrowid
            except sqlite3.Error:
                logger.warning("Failed to add episode", exc_info=True)
                return None

    def get_episode(self, episode_id: int) -> Episode | None:
        """Retrieve a single episode by ID."""
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT id, text, timestamp, session_id, importance, "
                    "last_cited_at, citation_count, embedding "
                    "FROM episodes WHERE id = ?",
                    (episode_id,),
                ).fetchone()
                if row is None:
                    return None
                return self._row_to_episode(row)
            except sqlite3.Error:
                logger.warning("Failed to get episode %d", episode_id, exc_info=True)
                return None

    def get_episodes_by_ids(self, ids: list[int]) -> list[Episode]:
        """Retrieve multiple episodes by ID."""
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            try:
                rows = self._conn.execute(
                    f"SELECT id, text, timestamp, session_id, importance, "
                    f"last_cited_at, citation_count, embedding "
                    f"FROM episodes WHERE id IN ({placeholders})",
                    ids,
                ).fetchall()
                return [self._row_to_episode(row) for row in rows]
            except sqlite3.Error:
                logger.warning("Failed to get episodes by ids", exc_info=True)
                return []

    def get_episodes_by_session_ids(self, session_ids: list[str]) -> dict[str, list[Episode]]:
        """Retrieve episodes grouped by session ID."""
        if not session_ids:
            return {}
        placeholders = ",".join("?" for _ in session_ids)
        with self._lock:
            try:
                rows = self._conn.execute(
                    f"SELECT id, text, timestamp, session_id, importance, "
                    f"last_cited_at, citation_count, embedding "
                    f"FROM episodes WHERE session_id IN ({placeholders}) "
                    f"ORDER BY session_id, timestamp",
                    session_ids,
                ).fetchall()
                result: dict[str, list[Episode]] = {}
                for row in rows:
                    ep = self._row_to_episode(row)
                    result.setdefault(ep.session_id, []).append(ep)
                return result
            except sqlite3.Error:
                logger.warning("Failed to get episodes by session ids", exc_info=True)
                return {}

    def update_episode_cited(self, episode_id: int, cited_at: str) -> None:
        """Update the last_cited_at timestamp for an episode."""
        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE episodes SET last_cited_at = ? WHERE id = ?",
                    (cited_at, episode_id),
                )
                self._conn.commit()
            except sqlite3.Error:
                logger.warning("Failed to update episode %d cited_at", episode_id, exc_info=True)

    def update_episode_embedding(self, episode_id: int, embedding: np.ndarray) -> None:
        """Update the embedding vector for an episode."""
        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE episodes SET embedding = ? WHERE id = ?",
                    (embedding.astype(np.float32).tobytes(), episode_id),
                )
                self._conn.commit()
            except sqlite3.Error:
                logger.warning("Failed to update episode %d embedding", episode_id, exc_info=True)

    def search_bm25(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """BM25 search over episodes using FTS5."""
        safe_query = self._sanitize_fts_query(query)
        if not safe_query:
            return []
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT rowid, rank FROM episodes_fts WHERE episodes_fts MATCH ? ORDER BY rank LIMIT ?",
                    (safe_query, top_k),
                ).fetchall()
                # FTS5 rank is negative (more negative = better). Negate for
                # a positive score where higher = better.
                return [(row[0], -row[1]) for row in rows]
            except sqlite3.Error:
                logger.warning("BM25 search failed", exc_info=True)
                return []

    # --- Profile ---

    def get_all_profiles(self) -> list[Profile]:
        """Load all user profile slots."""
        with self._lock:
            try:
                rows = self._conn.execute("SELECT id, topic, sub_topic, content, updated_at FROM profiles").fetchall()
                return [
                    Profile(
                        id=row[0],
                        topic=row[1],
                        sub_topic=row[2],
                        content=row[3],
                        updated_at=row[4],
                    )
                    for row in rows
                ]
            except sqlite3.Error:
                logger.warning("Failed to load profiles", exc_info=True)
                return []

    def upsert_profile(self, profile: Profile) -> int | None:
        """Insert or update a profile slot."""
        with self._lock:
            try:
                if profile.id is not None:
                    self._conn.execute(
                        "UPDATE profiles SET topic = ?, sub_topic = ?, content = ?, updated_at = ? WHERE id = ?",
                        (
                            profile.topic,
                            profile.sub_topic,
                            profile.content,
                            profile.updated_at,
                            profile.id,
                        ),
                    )
                    self._conn.commit()
                    return profile.id
                else:
                    cursor = self._conn.execute(
                        "INSERT INTO profiles (topic, sub_topic, content, updated_at) VALUES (?, ?, ?, ?)",
                        (profile.topic, profile.sub_topic, profile.content, profile.updated_at),
                    )
                    self._conn.commit()
                    return cursor.lastrowid
            except sqlite3.Error:
                logger.warning("Failed to upsert profile", exc_info=True)
                return None

    def delete_profile(self, profile_id: int) -> None:
        """Delete a profile slot."""
        with self._lock:
            try:
                self._conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
                self._conn.commit()
            except sqlite3.Error:
                logger.warning("Failed to delete profile %d", profile_id, exc_info=True)

    # --- Utterance ---

    def add_utterance(self, session_id: str, role: str, text: str, timestamp: str, token_count: int = 0) -> None:
        """Store a conversation utterance."""
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO utterances (session_id, role, text, timestamp, token_count) VALUES (?, ?, ?, ?, ?)",
                    (session_id, role, text, timestamp, token_count),
                )
                self._conn.commit()
            except sqlite3.Error:
                logger.warning("Failed to add utterance", exc_info=True)

    def get_utterances(self, session_id: str) -> list[tuple[str, str, str, int]]:
        """Retrieve all utterances for a session."""
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT role, text, timestamp, token_count FROM utterances "
                    "WHERE session_id = ? ORDER BY timestamp, id",
                    (session_id,),
                ).fetchall()
                return [(row[0], row[1], row[2], row[3]) for row in rows]
            except sqlite3.Error:
                logger.warning("Failed to get utterances for session %s", session_id, exc_info=True)
                return []

    def get_recent_sessions(
        self,
        limit: int,
        exclude_session_id: str | None = None,
    ) -> list[tuple[str, str]]:
        """Return recent sessions based on utterance timestamps."""
        with self._lock:
            try:
                if exclude_session_id:
                    rows = self._conn.execute(
                        "SELECT session_id, MIN(timestamp) as started_at "
                        "FROM utterances WHERE session_id != ? "
                        "GROUP BY session_id ORDER BY started_at DESC LIMIT ?",
                        (exclude_session_id, limit),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT session_id, MIN(timestamp) as started_at "
                        "FROM utterances "
                        "GROUP BY session_id ORDER BY started_at DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                return [(row[0], row[1]) for row in rows]
            except sqlite3.Error:
                logger.warning("Failed to get recent sessions", exc_info=True)
                return []

    # --- Session processing status ---

    def mark_session_processed(self, session_id: str) -> None:
        """Record that memory extraction has been attempted for a session."""
        with self._lock:
            try:
                from datetime import datetime

                now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
                self._conn.execute(
                    "INSERT OR IGNORE INTO processed_sessions (session_id, processed_at) VALUES (?, ?)",
                    (session_id, now),
                )
                self._conn.commit()
            except sqlite3.Error:
                logger.warning("Failed to mark session %s as processed", session_id, exc_info=True)

    def get_processed_session_ids(self, session_ids: list[str]) -> set[str]:
        """Check which sessions have been processed."""
        if not session_ids:
            return set()
        with self._lock:
            try:
                placeholders = ",".join("?" for _ in session_ids)
                rows = self._conn.execute(
                    f"SELECT session_id FROM processed_sessions WHERE session_id IN ({placeholders})",
                    session_ids,
                ).fetchall()
                return {row[0] for row in rows}
            except sqlite3.Error:
                logger.warning("Failed to get processed session IDs", exc_info=True)
                return set()

    # --- Lifecycle ---

    def load_all_embeddings(self) -> tuple[list[int], np.ndarray]:
        """Load all episode embeddings for vector index initialization."""
        with self._lock:
            try:
                rows = self._conn.execute("SELECT id, embedding FROM episodes WHERE embedding IS NOT NULL").fetchall()
                if not rows:
                    return [], np.empty((0, self._dimension), dtype=np.float32)
                expected_bytes = self._dimension * 4  # float32
                ids = []
                vecs = []
                for row in rows:
                    if len(row[1]) != expected_bytes:
                        logger.warning(
                            "Skipping episode %d: embedding size %d != expected %d",
                            row[0],
                            len(row[1]),
                            expected_bytes,
                        )
                        continue
                    ids.append(row[0])
                    vecs.append(np.frombuffer(row[1], dtype=np.float32).copy())
                if not ids:
                    return [], np.empty((0, self._dimension), dtype=np.float32)
                return ids, np.stack(vecs)
            except (sqlite3.Error, ValueError):
                logger.warning("Failed to load embeddings", exc_info=True)
                return [], np.empty((0, self._dimension), dtype=np.float32)

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                logger.debug("Error closing memory DB connection", exc_info=True)

    # --- Internal ---

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize a query string for FTS5 MATCH.

        Wraps each token in double quotes to prevent FTS5 special
        characters (-, *, OR, AND, NOT, NEAR, etc.) from being
        interpreted as operators.
        """
        tokens = query.split()
        if not tokens:
            return ""
        safe_tokens = ['"' + t.replace('"', '""') + '"' for t in tokens]
        return " ".join(safe_tokens)

    def _row_to_episode(self, row: Any) -> Episode:
        embedding = np.frombuffer(row[7], dtype=np.float32).copy() if row[7] is not None else None
        return Episode(
            id=row[0],
            text=row[1],
            timestamp=row[2],
            session_id=row[3],
            importance=row[4],
            last_cited_at=row[5],
            citation_count=row[6],
            embedding=embedding,
        )


class InMemoryMemoryStorage(IMemoryStorage):
    """In-memory storage backend for unit tests.

    Simple list-based implementation. BM25 search is approximated
    by case-insensitive word overlap counting.
    """

    def __init__(self, dimension: int = _DEFAULT_DIMENSION) -> None:
        self._dimension = dimension
        self._episodes: dict[int, Episode] = {}
        self._profiles: dict[int, Profile] = {}
        self._utterances: list[dict[str, Any]] = []
        self._processed: set[str] = set()
        self._next_episode_id = 1
        self._next_profile_id = 1

    # --- Episode ---

    def add_episode(self, episode: Episode) -> int | None:
        """Persist a new episode."""
        eid = self._next_episode_id
        self._next_episode_id += 1
        stored = Episode(
            id=eid,
            text=episode.text,
            timestamp=episode.timestamp,
            session_id=episode.session_id,
            importance=episode.importance,
            last_cited_at=episode.last_cited_at,
            citation_count=episode.citation_count,
            embedding=episode.embedding.copy() if episode.embedding is not None else None,
        )
        self._episodes[eid] = stored
        return eid

    def get_episode(self, episode_id: int) -> Episode | None:
        """Retrieve a single episode by ID."""
        ep = self._episodes.get(episode_id)
        if ep is None:
            return None
        return Episode(
            id=ep.id,
            text=ep.text,
            timestamp=ep.timestamp,
            session_id=ep.session_id,
            importance=ep.importance,
            last_cited_at=ep.last_cited_at,
            citation_count=ep.citation_count,
            embedding=ep.embedding.copy() if ep.embedding is not None else None,
        )

    def get_episodes_by_ids(self, ids: list[int]) -> list[Episode]:
        """Retrieve multiple episodes by ID."""
        results = []
        for eid in ids:
            ep = self.get_episode(eid)
            if ep is not None:
                results.append(ep)
        return results

    def get_episodes_by_session_ids(self, session_ids: list[str]) -> dict[str, list[Episode]]:
        """Retrieve episodes grouped by session ID."""
        sid_set = set(session_ids)
        result: dict[str, list[Episode]] = {}
        for ep in self._episodes.values():
            if ep.session_id in sid_set:
                copy = Episode(
                    id=ep.id,
                    text=ep.text,
                    timestamp=ep.timestamp,
                    session_id=ep.session_id,
                    importance=ep.importance,
                    last_cited_at=ep.last_cited_at,
                    citation_count=ep.citation_count,
                    embedding=ep.embedding.copy() if ep.embedding is not None else None,
                )
                result.setdefault(ep.session_id, []).append(copy)
        for episodes in result.values():
            episodes.sort(key=lambda e: e.timestamp)
        return result

    def update_episode_cited(self, episode_id: int, cited_at: str) -> None:
        """Update the last_cited_at timestamp for an episode."""
        ep = self._episodes.get(episode_id)
        if ep is not None:
            ep.last_cited_at = cited_at

    def update_episode_embedding(self, episode_id: int, embedding: np.ndarray) -> None:
        """Update the embedding vector for an episode."""
        ep = self._episodes.get(episode_id)
        if ep is not None:
            ep.embedding = embedding.astype(np.float32).copy()

    def search_bm25(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Approximate BM25 via word overlap counting."""
        query_words = set(query.lower().split())
        if not query_words:
            return []
        scored: list[tuple[int, float]] = []
        for eid, ep in self._episodes.items():
            doc_words = set(ep.text.lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                scored.append((eid, float(overlap)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # --- Profile ---

    def get_all_profiles(self) -> list[Profile]:
        """Load all user profile slots."""
        return [
            Profile(
                id=p.id,
                topic=p.topic,
                sub_topic=p.sub_topic,
                content=p.content,
                updated_at=p.updated_at,
            )
            for p in self._profiles.values()
        ]

    def upsert_profile(self, profile: Profile) -> int | None:
        """Insert or update a profile slot."""
        if profile.id is not None and profile.id in self._profiles:
            self._profiles[profile.id] = Profile(
                id=profile.id,
                topic=profile.topic,
                sub_topic=profile.sub_topic,
                content=profile.content,
                updated_at=profile.updated_at,
            )
            return profile.id
        pid = self._next_profile_id
        self._next_profile_id += 1
        self._profiles[pid] = Profile(
            id=pid,
            topic=profile.topic,
            sub_topic=profile.sub_topic,
            content=profile.content,
            updated_at=profile.updated_at,
        )
        return pid

    def delete_profile(self, profile_id: int) -> None:
        """Delete a profile slot."""
        self._profiles.pop(profile_id, None)

    # --- Utterance ---

    def add_utterance(self, session_id: str, role: str, text: str, timestamp: str, token_count: int = 0) -> None:
        """Store a conversation utterance."""
        self._utterances.append(
            {
                "session_id": session_id,
                "role": role,
                "text": text,
                "timestamp": timestamp,
                "token_count": token_count,
            }
        )

    def get_utterances(self, session_id: str) -> list[tuple[str, str, str, int]]:
        """Retrieve all utterances for a session."""
        return [
            (u["role"], u["text"], u["timestamp"], u["token_count"])
            for u in self._utterances
            if u["session_id"] == session_id
        ]

    def get_recent_sessions(
        self,
        limit: int,
        exclude_session_id: str | None = None,
    ) -> list[tuple[str, str]]:
        """Return recent sessions based on utterance timestamps."""
        sessions: dict[str, str] = {}
        for u in self._utterances:
            sid = u["session_id"]
            if sid == exclude_session_id:
                continue
            if sid not in sessions or u["timestamp"] < sessions[sid]:
                sessions[sid] = u["timestamp"]
        ordered = sorted(sessions.items(), key=lambda x: x[1], reverse=True)
        return ordered[:limit]

    # --- Session processing status ---

    def mark_session_processed(self, session_id: str) -> None:
        """Record that memory extraction has been attempted."""
        self._processed.add(session_id)

    def get_processed_session_ids(self, session_ids: list[str]) -> set[str]:
        """Check which sessions have been processed."""
        return self._processed & set(session_ids)

    # --- Lifecycle ---

    def load_all_embeddings(self) -> tuple[list[int], np.ndarray]:
        """Load all episode embeddings."""
        ids = []
        vecs = []
        for eid, ep in self._episodes.items():
            if ep.embedding is not None:
                ids.append(eid)
                vecs.append(ep.embedding)
        if not ids:
            return [], np.empty((0, self._dimension), dtype=np.float32)
        return ids, np.stack(vecs).astype(np.float32)

    def close(self) -> None:
        """No-op for in-memory backend."""
