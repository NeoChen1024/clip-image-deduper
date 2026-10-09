#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SQLite-backed embedding store for clip_image_deduper.

One database file holds every embedding of an image directory. Each row is keyed by the image's relative
path and stores the image mtime observed at encoding time plus the raw float16 embedding bytes. (CLIP models run in
fp16 on CUDA, so their outputs are exactly representable in fp16 and storing them as fp32 just doubles the size.) A small
``meta`` table records the model id and embedding dimension so a database is never silently reused with a
different model.
"""

import os
import sqlite3
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

SCHEMA_VERSION = "1"
EMBEDDING_DTYPE = np.float16

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    path      TEXT PRIMARY KEY,
    mtime     REAL NOT NULL,
    size      INTEGER NOT NULL,
    embedding BLOB NOT NULL
);
"""
# Note: deliberately a rowid table. WITHOUT ROWID stores rows in an index b-tree whose small local payload
# limit pushes every ~5 KiB embedding into two overflow pages (~60% file bloat and 4x slower writes).


class ModelMismatchError(RuntimeError):
    """Raised when a database was encoded with a different model than requested."""


class EmbeddingDB:
    """Thin wrapper around a SQLite connection holding fixed-dimension embeddings."""

    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # -- context manager -------------------------------------------------------------------------------------------

    def __enter__(self) -> "EmbeddingDB":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # -- meta ------------------------------------------------------------------------------------------------------

    def get_meta(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(value)))

    @property
    def model_id(self) -> Optional[str]:
        return self.get_meta("model_id")

    @property
    def dim(self) -> Optional[int]:
        value = self.get_meta("dim")
        return None if value is None else int(value)

    def ensure_model(self, model_id: str, reset_on_mismatch: bool = False) -> None:
        """Bind the database to ``model_id``.

        A fresh database adopts the model. An existing database must match, unless ``reset_on_mismatch`` is set,
        in which case all rows are dropped and the database is re-bound (used by ``--force-update``).
        """
        current = self.model_id
        if current is None:
            self.set_meta("schema_version", SCHEMA_VERSION)
            self.set_meta("model_id", model_id)
            self.set_meta("dtype", np.dtype(EMBEDDING_DTYPE).name)
            self.conn.commit()
            return
        if current == model_id:
            return
        if not reset_on_mismatch:
            raise ModelMismatchError(
                f"Database {self.path} was encoded with model '{current}', but '{model_id}' was requested. "
                "Pass --force-update to re-encode everything with the new model, or use a different database file."
            )
        self.conn.execute("DELETE FROM embeddings")
        self.conn.execute("DELETE FROM meta")
        self.set_meta("schema_version", SCHEMA_VERSION)
        self.set_meta("model_id", model_id)
        self.set_meta("dtype", np.dtype(EMBEDDING_DTYPE).name)
        self.conn.commit()

    # -- rows ------------------------------------------------------------------------------------------------------

    def __len__(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    def load_index(self) -> Dict[str, float]:
        """Return ``{relative_path: mtime}`` for every stored embedding."""
        return dict(self.conn.execute("SELECT path, mtime FROM embeddings"))

    def upsert(self, rows: Iterable[Tuple[str, float, int, np.ndarray]]) -> None:
        """Insert or replace ``(path, mtime, size, embedding)`` rows in a single transaction."""
        prepared = []
        for path, mtime, size, embedding in rows:
            vec = np.ascontiguousarray(embedding, dtype=EMBEDDING_DTYPE).reshape(-1)
            dim = self.dim
            if dim is None:
                self.set_meta("dim", str(vec.shape[0]))
            elif vec.shape[0] != dim:
                raise ValueError(f"Embedding for {path} has dimension {vec.shape[0]}, database expects {dim}")
            prepared.append((path, float(mtime), int(size), vec.tobytes()))
        if prepared:
            self.conn.executemany(
                "INSERT OR REPLACE INTO embeddings (path, mtime, size, embedding) VALUES (?, ?, ?, ?)", prepared
            )
        self.conn.commit()

    def delete(self, paths: Iterable[str]) -> int:
        """Delete rows by path; returns the number of rows removed."""
        paths = list(paths)
        if not paths:
            return 0
        cur = self.conn.executemany("DELETE FROM embeddings WHERE path = ?", ((p,) for p in paths))
        self.conn.commit()
        return cur.rowcount if cur.rowcount >= 0 else len(paths)

    def load_all(self) -> Tuple[List[str], np.ndarray]:
        """Return every embedding as ``(paths, (N, D) float16 array)`` with one memory copy."""
        dim = self.dim
        rows = self.conn.execute("SELECT path, embedding FROM embeddings ORDER BY path").fetchall()
        if not rows:
            return [], np.empty((0, dim or 0), dtype=EMBEDDING_DTYPE)
        if dim is None:
            raise RuntimeError(f"Database {self.path} has rows but no recorded dimension; it is corrupt.")
        paths = [row[0] for row in rows]
        # bytearray.join yields a mutable buffer, so frombuffer below is a writable zero-copy view.
        blob = bytearray().join(row[1] for row in rows)
        expected = len(rows) * dim * np.dtype(EMBEDDING_DTYPE).itemsize
        if len(blob) != expected:
            raise RuntimeError(f"Database {self.path} embedding blobs total {len(blob)} bytes, expected {expected}; it is corrupt.")
        embeddings = np.frombuffer(blob, dtype=EMBEDDING_DTYPE).reshape(len(rows), dim)
        return paths, embeddings
