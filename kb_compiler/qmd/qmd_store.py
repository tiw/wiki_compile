"""QmdIndexStore: SQLite + FTS5 + sqlite-vec hybrid search storage."""

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from kb_compiler.qmd.chunker import Chunk

_SQLITE_VEC_AVAILABLE = False
try:
    import sqlite_vec

    _SQLITE_VEC_AVAILABLE = True
except ImportError:
    sqlite_vec = None  # type: ignore


@dataclass
class SearchResult:
    """Result from hybrid search."""

    chunk_id: str
    source_path: str
    concept_name: str
    section_header: Optional[str]
    content: str
    token_count: int
    score: float


class QmdIndexStore:
    """Manages SQLite, FTS5, and sqlite-vec indices for qmd."""

    def __init__(self, db_path: Path, embedding_dim: int):
        if not _SQLITE_VEC_AVAILABLE:
            raise ImportError(
                "sqlite-vec is not installed. "
                "Install qmd extras with: pip install 'kb-compiler[qmd]'"
            )

        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._conn = None

    def _get_conn(self):
        if self._conn is None:
            import sqlite3

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            if hasattr(self._conn, "enable_load_extension"):
                self._conn.enable_load_extension(True)
                sqlite_vec.load(self._conn)  # type: ignore
                self._conn.enable_load_extension(False)
            else:
                raise RuntimeError(
                    "Your Python build does not support SQLite extension loading. "
                    "Please use a Python compiled with loadable extension support "
                    "(e.g., Homebrew Python on macOS: /opt/homebrew/bin/python3)."
                )
        return self._conn

    def rebuild(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Rebuild the entire index from chunks and embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
            )

        conn = self._get_conn()
        cursor = conn.cursor()

        # Drop existing indices
        cursor.execute("DROP TABLE IF EXISTS fts_chunks")
        cursor.execute("DROP TABLE IF EXISTS vec_chunks")
        cursor.execute("DROP TABLE IF EXISTS chunks")

        # Create base table
        cursor.execute(
            """
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE,
                source_path TEXT,
                concept_name TEXT,
                section_header TEXT,
                content TEXT,
                token_count INTEGER
            )
            """
        )

        # Create FTS5 external content table
        cursor.execute(
            """
            CREATE VIRTUAL TABLE fts_chunks USING fts5(
                content,
                content='chunks',
                content_rowid='id'
            )
            """
        )

        # Create vec0 virtual table
        cursor.execute(
            f"""
            CREATE VIRTUAL TABLE vec_chunks USING vec0(
                row_id INTEGER PRIMARY KEY,
                embedding float[{self.embedding_dim}]
            )
            """
        )

        # Insert chunks
        chunk_rows = [
            (
                c.id,
                c.source_path,
                c.concept_name,
                c.section_header,
                c.content,
                c.token_count,
            )
            for c in chunks
        ]
        cursor.executemany(
            """
            INSERT INTO chunks (chunk_id, source_path, concept_name, section_header, content, token_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            chunk_rows,
        )

        # Retrieve rowids in insertion order to match embeddings
        cursor.execute("SELECT id FROM chunks ORDER BY id")
        rowids = [row[0] for row in cursor.fetchall()]

        # Insert embeddings into vec_chunks
        vec_rows = [
            (row_id, _serialize_embedding(emb))
            for row_id, emb in zip(rowids, embeddings)
        ]
        cursor.executemany(
            "INSERT INTO vec_chunks (row_id, embedding) VALUES (?, ?)",
            vec_rows,
        )

        # Populate FTS5 index
        cursor.execute(
            "INSERT INTO fts_chunks(rowid, content) SELECT id, content FROM chunks"
        )
        cursor.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('optimize')")

        conn.commit()

    def search_hybrid(
        self,
        query: str,
        query_embedding: list[float],
        k: int = 20,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """Search using BM25 + Vector + RRF fusion."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Serialize query embedding
        query_vec = _serialize_embedding(query_embedding)

        sql = """
        WITH
        fts_matches AS (
            SELECT rowid AS id, rank AS fts_score
            FROM fts_chunks
            WHERE content MATCH ?
            ORDER BY rank
            LIMIT ?
        ),
        vec_matches AS (
            SELECT row_id AS id, distance AS vec_score
            FROM vec_chunks
            WHERE embedding MATCH ?
              AND k = ?
            ORDER BY distance
            LIMIT ?
        ),
        fts_ranked AS (
            SELECT id, fts_score,
                   row_number() OVER (ORDER BY fts_score) AS fts_rank
            FROM fts_matches
        ),
        vec_ranked AS (
            SELECT id, vec_score,
                   row_number() OVER (ORDER BY vec_score) AS vec_rank
            FROM vec_matches
        ),
        combined AS (
            SELECT id, fts_rank, NULL AS vec_rank,
                   1.0 / (? + fts_rank) AS rrf_score
            FROM fts_ranked
            UNION ALL
            SELECT id, NULL AS fts_rank, vec_rank,
                   1.0 / (? + vec_rank) AS rrf_score
            FROM vec_ranked
        ),
        grouped AS (
            SELECT id,
                   sum(rrf_score) AS rrf_score,
                   max(fts_rank) AS fts_rank,
                   max(vec_rank) AS vec_rank
            FROM combined
            GROUP BY id
            ORDER BY rrf_score DESC
            LIMIT ?
        )
        SELECT c.chunk_id, c.source_path, c.concept_name, c.section_header,
               c.content, c.token_count, g.rrf_score
        FROM grouped g
        JOIN chunks c ON c.id = g.id
        ORDER BY g.rrf_score DESC
        """

        params = (
            query,
            k,
            query_vec,
            k,
            k,
            rrf_k,
            rrf_k,
            k,
        )

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        return [
            SearchResult(
                chunk_id=row["chunk_id"],
                source_path=row["source_path"],
                concept_name=row["concept_name"],
                section_header=row["section_header"],
                content=row["content"],
                token_count=row["token_count"],
                score=row["rrf_score"],
            )
            for row in rows
        ]

    def stats(self) -> dict:
        """Return index statistics."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM chunks")
        total_chunks = cursor.fetchone()[0]
        cursor.execute("SELECT count(DISTINCT concept_name) FROM chunks")
        total_concepts = cursor.fetchone()[0]
        return {
            "total_chunks": total_chunks,
            "total_concepts": total_concepts,
            "embedding_dim": self.embedding_dim,
            "db_path": str(self.db_path),
        }


def _serialize_embedding(vector: list[float]) -> bytes:
    """Serialize a float vector into bytes for sqlite-vec."""
    if sqlite_vec is not None:
        # Prefer sqlite_vec helpers if available
        for attr in ("serialize_float32", "serialize_f32"):
            fn = getattr(sqlite_vec, attr, None)
            if fn is not None:
                return fn(vector)  # type: ignore
    return struct.pack(f"{len(vector)}f", *vector)
