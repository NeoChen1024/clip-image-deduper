"""Review sessions: duplicate groups found at a loose threshold, the reviewer's decisions, and applying them.

Everything lives in a SQLite file of its own next to the embedding database (``pictures.review.sqlite`` for
``pictures.sqlite``), so a ``--force-update`` rebuild of the embeddings never touches decisions, and a review of a
large library (tens of thousands of groups) is indexed rather than parsed. No Qt in here; the GUI and the CLI both
drive this module.

Group identity is the hash of its sorted member paths. Re-matching upserts by that key: an unchanged group keeps
its row, decision and history; a group whose membership changed is a new group, and the old row is marked
``stale``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from .db_store import ImageRecord
from .dedupe import DuplicateGroup, move_to_trash
from .keeping import Policy

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1"
STATUSES = ("pending", "decided", "skipped", "applied", "stale")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS groups (
    id           INTEGER PRIMARY KEY,
    key          TEXT NOT NULL UNIQUE,
    n            INTEGER NOT NULL,
    min_distance REAL NOT NULL,
    max_distance REAL NOT NULL,
    status       TEXT NOT NULL,
    note         TEXT NOT NULL DEFAULT '',
    updated_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS groups_status ON groups (status, min_distance);
CREATE TABLE IF NOT EXISTS members (
    group_id           INTEGER NOT NULL REFERENCES groups (id) ON DELETE CASCADE,
    path               TEXT NOT NULL,
    keep               INTEGER NOT NULL,
    width              INTEGER NOT NULL,
    height             INTEGER NOT NULL,
    size               INTEGER NOT NULL,
    format             TEXT NOT NULL,
    mtime              REAL NOT NULL,
    distance_to_winner REAL,
    PRIMARY KEY (group_id, path)
);
CREATE INDEX IF NOT EXISTS members_path ON members (path);
CREATE TABLE IF NOT EXISTS edges (
    group_id INTEGER NOT NULL REFERENCES groups (id) ON DELETE CASCADE,
    path_a   TEXT NOT NULL,
    path_b   TEXT NOT NULL,
    distance REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS edges_group ON edges (group_id);
CREATE TABLE IF NOT EXISTS history (
    id        INTEGER PRIMARY KEY,
    group_id  INTEGER NOT NULL REFERENCES groups (id) ON DELETE CASCADE,
    at        REAL NOT NULL,
    keep_json TEXT NOT NULL,
    status    TEXT NOT NULL,
    note      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS history_group ON history (group_id, id);
CREATE TABLE IF NOT EXISTS applied (
    path       TEXT PRIMARY KEY,
    trash_path TEXT NOT NULL,
    group_id   INTEGER NOT NULL REFERENCES groups (id) ON DELETE CASCADE,
    at         REAL NOT NULL
);
"""


def default_review_path(db_path: str) -> str:
    """``pictures.sqlite`` -> ``pictures.review.sqlite``."""
    root, _ = os.path.splitext(db_path)
    return root + ".review.sqlite"


def group_key(paths: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(paths)).encode("utf-8")).hexdigest()


@dataclass(slots=True, frozen=True)
class GroupRow:
    id: int
    key: str
    n: int
    min_distance: float
    max_distance: float
    status: str
    note: str
    updated_at: float


@dataclass(slots=True, frozen=True)
class MemberRow:
    path: str
    keep: bool
    width: int
    height: int
    size: int
    format: str
    mtime: float
    distance_to_winner: float | None

    @property
    def record(self) -> ImageRecord:
        return ImageRecord(self.path, self.mtime, self.size, self.width, self.height, self.format)


@dataclass(slots=True)
class UpsertStats:
    new: int = 0
    kept: int = 0
    stale: int = 0
    predecided: int = 0


@dataclass(slots=True, frozen=True)
class Move:
    group_id: int
    path: str
    keep: tuple[str, ...]


class ReviewDB:
    """The review session file. Every decision is committed as it is made."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", SCHEMA_VERSION)
        self.conn.commit()

    def __enter__(self) -> ReviewDB:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # -- meta ------------------------------------------------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def set_meta(self, key: str, value: object) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(value)))

    def begin_session(self, *, image_dir: str, model_id: str, review_threshold: float, auto_threshold: float, policy: str) -> None:
        """Record what the groups were produced with. A later session with another model refuses to mix."""
        previous = self.get_meta("model_id")
        if previous is not None and previous != model_id:
            raise ValueError(f"Review database {self.path} holds groups from model '{previous}', not '{model_id}'. Use another file.")
        for k, v in (
            ("image_dir", os.path.abspath(image_dir)),
            ("model_id", model_id),
            ("review_threshold", review_threshold),
            ("auto_threshold", auto_threshold),
            ("policy", policy),
        ):
            self.set_meta(k, v)
        if self.get_meta("created_at") is None:
            self.set_meta("created_at", time.time())
        self.conn.commit()

    @property
    def review_threshold(self) -> float | None:
        v = self.get_meta("review_threshold")
        return None if v is None else float(v)

    @property
    def auto_threshold(self) -> float | None:
        v = self.get_meta("auto_threshold")
        return None if v is None else float(v)

    # -- matching results --------------------------------------------------------------------------------------------

    def upsert_groups(
        self, groups: Iterable[tuple[Sequence[ImageRecord], Sequence[tuple[str, str, float]]]], policy: Policy, auto_threshold: float
    ) -> UpsertStats:
        """Store the groups of one matching run.

        Each item is ``(member records, edges as (path_a, path_b, distance))``. New groups get the policy's winner
        as the only kept member; groups whose every edge is at or below ``auto_threshold`` start ``decided`` (that
        is what ``dedupe`` would have merged without asking). Existing groups keep their decision; file metadata is
        refreshed. Groups not in this run that were still open are marked ``stale``.
        """
        stats = UpsertStats()
        now = time.time()
        seen: set[str] = set()
        for records, edges in groups:
            key = group_key(r.path for r in records)
            seen.add(key)
            if not edges:
                raise ValueError("A group needs at least one edge")
            dmin = min(d for _, _, d in edges)
            dmax = max(d for _, _, d in edges)
            winner = policy.select(records)
            by_pair = {}
            for a, b, d in edges:
                by_pair[(a, b)] = d
                by_pair[(b, a)] = d
            row = self.conn.execute("SELECT id, status FROM groups WHERE key = ?", (key,)).fetchone()
            if row is None:
                status = "decided" if dmax <= auto_threshold else "pending"
                cur = self.conn.execute(
                    "INSERT INTO groups (key, n, min_distance, max_distance, status, note, updated_at) VALUES (?, ?, ?, ?, ?, '', ?)",
                    (key, len(records), dmin, dmax, status, now),
                )
                assert cur.lastrowid is not None
                gid = int(cur.lastrowid)
                self.conn.executemany(
                    "INSERT INTO members (group_id, path, keep, width, height, size, format, mtime, distance_to_winner) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (gid, r.path, int(r is winner), r.width, r.height, r.size, r.format, r.mtime, 0.0 if r is winner else by_pair.get((r.path, winner.path)))
                        for r in records
                    ],
                )
                self._snapshot(gid, now)
                stats.new += 1
                if status == "decided":
                    stats.predecided += 1
            else:
                gid = int(row["id"])
                new_status = row["status"]
                if new_status == "stale":
                    new_status = "pending"
                self.conn.execute(
                    "UPDATE groups SET n = ?, min_distance = ?, max_distance = ?, status = ?, updated_at = ? WHERE id = ?",
                    (len(records), dmin, dmax, new_status, now, gid),
                )
                for r in records:
                    self.conn.execute(
                        "UPDATE members SET width = ?, height = ?, size = ?, format = ?, mtime = ?, distance_to_winner = ? WHERE group_id = ? AND path = ?",
                        (r.width, r.height, r.size, r.format, r.mtime, 0.0 if r is winner else by_pair.get((r.path, winner.path)), gid, r.path),
                    )
                self.conn.execute("DELETE FROM edges WHERE group_id = ?", (gid,))
                stats.kept += 1
            self.conn.executemany("INSERT INTO edges (group_id, path_a, path_b, distance) VALUES (?, ?, ?, ?)", [(gid, a, b, d) for a, b, d in edges])
        open_rows = self.conn.execute("SELECT id, key FROM groups WHERE status IN ('pending', 'decided', 'skipped')").fetchall()
        for r in open_rows:
            if r["key"] not in seen:
                self.conn.execute("UPDATE groups SET status = 'stale', updated_at = ? WHERE id = ?", (now, r["id"]))
                stats.stale += 1
        self.conn.commit()
        return stats

    # -- reading -----------------------------------------------------------------------------------------------------

    @staticmethod
    def _group(row: sqlite3.Row) -> GroupRow:
        return GroupRow(row["id"], row["key"], row["n"], row["min_distance"], row["max_distance"], row["status"], row["note"], row["updated_at"])

    def groups(self, status: str | Sequence[str] | None = None) -> list[GroupRow]:
        """Groups ordered by minimum distance (the surest first). ``status`` filters; ``None`` means every status
        except ``stale``."""
        if status is None:
            statuses: tuple[str, ...] = ("pending", "decided", "skipped", "applied")
        elif isinstance(status, str):
            statuses = (status,)
        else:
            statuses = tuple(status)
        marks = ",".join("?" * len(statuses))
        rows = self.conn.execute(f"SELECT * FROM groups WHERE status IN ({marks}) ORDER BY min_distance, id", statuses).fetchall()
        return [self._group(r) for r in rows]

    def group(self, group_id: int) -> GroupRow:
        row = self.conn.execute("SELECT * FROM groups WHERE id = ?", (group_id,)).fetchone()
        if row is None:
            raise KeyError(group_id)
        return self._group(row)

    def counts(self) -> dict[str, int]:
        out = {s: 0 for s in STATUSES}
        for r in self.conn.execute("SELECT status, COUNT(*) AS c FROM groups GROUP BY status"):
            out[r["status"]] = r["c"]
        return out

    def members(self, group_id: int) -> list[MemberRow]:
        rows = self.conn.execute("SELECT * FROM members WHERE group_id = ? ORDER BY path", (group_id,)).fetchall()
        return [MemberRow(r["path"], bool(r["keep"]), r["width"], r["height"], r["size"], r["format"], r["mtime"], r["distance_to_winner"]) for r in rows]

    def edges(self, group_id: int) -> list[tuple[str, str, float]]:
        rows = self.conn.execute("SELECT path_a, path_b, distance FROM edges WHERE group_id = ? ORDER BY distance", (group_id,)).fetchall()
        return [(r["path_a"], r["path_b"], r["distance"]) for r in rows]

    def groups_of(self, path: str) -> list[int]:
        return [r["group_id"] for r in self.conn.execute("SELECT group_id FROM members WHERE path = ?", (path,))]

    # -- deciding ----------------------------------------------------------------------------------------------------

    def _snapshot(self, group_id: int, now: float) -> None:
        keeps = {r["path"]: bool(r["keep"]) for r in self.conn.execute("SELECT path, keep FROM members WHERE group_id = ?", (group_id,))}
        g = self.conn.execute("SELECT status, note FROM groups WHERE id = ?", (group_id,)).fetchone()
        self.conn.execute(
            "INSERT INTO history (group_id, at, keep_json, status, note) VALUES (?, ?, ?, ?, ?)",
            (group_id, now, json.dumps(keeps, sort_keys=True), g["status"], g["note"]),
        )

    def decide(self, group_id: int, *, keeps: dict[str, bool] | None = None, status: str | None = None, note: str | None = None) -> None:
        """Change keep flags, status and/or note of a group in one committed step that ``undo`` can revert."""
        if status is not None and status not in STATUSES:
            raise ValueError(f"Unknown status {status!r}")
        g = self.group(group_id)
        if g.status == "applied":
            raise ValueError("An applied group cannot be changed; undo the apply first")
        now = time.time()
        if keeps:
            known = {m.path for m in self.members(group_id)}
            unknown = set(keeps) - known
            if unknown:
                raise KeyError(f"Not members of group {group_id}: {sorted(unknown)}")
            self.conn.executemany("UPDATE members SET keep = ? WHERE group_id = ? AND path = ?", [(int(v), group_id, p) for p, v in keeps.items()])
        sets: list[str] = ["updated_at = ?"]
        args: list[object] = [now]
        if status is not None:
            sets.append("status = ?")
            args.append(status)
        if note is not None:
            sets.append("note = ?")
            args.append(note)
        self.conn.execute(f"UPDATE groups SET {', '.join(sets)} WHERE id = ?", (*args, group_id))
        self._snapshot(group_id, now)
        self.conn.commit()

    def keep_only(self, group_id: int, path: str, *, status: str | None = "decided") -> None:
        keeps = {m.path: m.path == path for m in self.members(group_id)}
        if path not in keeps:
            raise KeyError(path)
        self.decide(group_id, keeps=keeps, status=status)

    def reset_to_policy(self, group_id: int, policy: Policy) -> None:
        members = self.members(group_id)
        winner = policy.select([m.record for m in members])
        self.decide(group_id, keeps={m.path: m.path == winner.path for m in members})

    def undo(self, group_id: int) -> bool:
        """Revert the last ``decide`` of a group. Returns False when there is nothing left to undo."""
        rows = self.conn.execute("SELECT id, keep_json, status, note FROM history WHERE group_id = ? ORDER BY id DESC LIMIT 2", (group_id,)).fetchall()
        if len(rows) < 2:
            return False
        current, previous = rows
        if self.group(group_id).status == "applied":
            raise ValueError("An applied group cannot be changed; undo the apply first")
        for path, keep in json.loads(previous["keep_json"]).items():
            self.conn.execute("UPDATE members SET keep = ? WHERE group_id = ? AND path = ?", (int(keep), group_id, path))
        self.conn.execute(
            "UPDATE groups SET status = ?, note = ?, updated_at = ? WHERE id = ?", (previous["status"], previous["note"], time.time(), group_id)
        )
        self.conn.execute("DELETE FROM history WHERE id = ?", (current["id"],))
        self.conn.commit()
        return True

    # -- applying ----------------------------------------------------------------------------------------------------

    def plan_apply(self, group_ids: Sequence[int] | None = None) -> list[Move]:
        """Files that applying the decided groups would move, with what is kept instead."""
        if group_ids is None:
            targets = [g.id for g in self.groups("decided")]
        else:
            targets = [gid for gid in group_ids if self.group(gid).status == "decided"]
        moves: list[Move] = []
        for gid in targets:
            members = self.members(gid)
            keep = tuple(m.path for m in members if m.keep)
            if not keep:
                logger.debug("Group %d keeps nothing; refusing to move all of its files", gid)
                continue
            moves.extend(Move(gid, m.path, keep) for m in members if not m.keep)
        return moves

    def mark_applied(self, group_id: int, moved: Iterable[tuple[str, str]]) -> None:
        now = time.time()
        self.conn.executemany(
            "INSERT OR REPLACE INTO applied (path, trash_path, group_id, at) VALUES (?, ?, ?, ?)", [(p, t, group_id, now) for p, t in moved]
        )
        self.conn.execute("UPDATE groups SET status = 'applied', updated_at = ? WHERE id = ?", (now, group_id))
        self._snapshot(group_id, now)
        self.conn.commit()

    def applied(self, group_id: int | None = None) -> list[tuple[str, str, int]]:
        sql = "SELECT path, trash_path, group_id FROM applied"
        rows = self.conn.execute(sql + (" WHERE group_id = ?" if group_id is not None else ""), (group_id,) if group_id is not None else ()).fetchall()
        return [(r["path"], r["trash_path"], r["group_id"]) for r in rows]

    def mark_unapplied(self, group_id: int) -> None:
        now = time.time()
        self.conn.execute("DELETE FROM applied WHERE group_id = ?", (group_id,))
        self.conn.execute("UPDATE groups SET status = 'decided', updated_at = ? WHERE id = ?", (now, group_id))
        self._snapshot(group_id, now)
        self.conn.commit()


def changed_on_disk(image_dir: str, members: Sequence[MemberRow]) -> list[str]:
    """Member paths whose file is missing or has a different mtime/size than when the group was stored."""
    out = []
    for m in members:
        try:
            st = os.stat(os.path.join(image_dir, m.path))
        except OSError:
            out.append(m.path)
            continue
        if st.st_mtime != m.mtime or st.st_size != m.size:
            out.append(m.path)
    return out


def apply_decisions(
    review: ReviewDB, image_dir: str, trash_dir: str, *, dry_run: bool, group_ids: Sequence[int] | None = None, progress: Callable[[int], object] | None = None
) -> tuple[int, list[int]]:
    """Move the losers of every decided group to ``trash_dir``. Returns ``(files moved, group ids refused)``.

    A group is refused, and left ``decided``, when any member changed on disk since it was stored; re-run
    ``update-db`` and re-match for those.
    """
    moves = review.plan_apply(group_ids)
    by_group: dict[int, list[Move]] = {}
    for mv in moves:
        by_group.setdefault(mv.group_id, []).append(mv)
    moved_total = 0
    refused: list[int] = []
    for gid, group_moves in by_group.items():
        members = review.members(gid)
        changed = changed_on_disk(image_dir, members)
        if changed:
            logger.warning("Group %d: %d file(s) changed on disk since matching, skipping: %s", gid, len(changed), ", ".join(changed[:5]))
            refused.append(gid)
            continue
        done: list[tuple[str, str]] = []
        for mv in group_moves:
            if move_to_trash(image_dir, mv.path, trash_dir, dry_run=dry_run, reason=f"keeping {', '.join(mv.keep)}"):
                done.append((mv.path, os.path.join(trash_dir, mv.path)))
        moved_total += len(done)
        if not dry_run:
            review.mark_applied(gid, done)
        if progress is not None:
            progress(1)
    return moved_total, refused


def unapply_group(review: ReviewDB, image_dir: str, group_id: int) -> int:
    """Move a group's trashed files back where they were. Returns the number restored."""
    restored = 0
    for path, trash_path, _ in review.applied(group_id):
        dest = os.path.join(image_dir, path)
        if not os.path.exists(trash_path):
            logger.warning("Cannot restore %s: %s is gone", path, trash_path)
            continue
        if os.path.exists(dest):
            logger.warning("Cannot restore %s: destination exists", path)
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.replace(trash_path, dest)
        restored += 1
    review.mark_unapplied(group_id)
    return restored


def store_matches(
    review: ReviewDB, groups: Sequence[DuplicateGroup], records: Sequence[ImageRecord], policy: Policy, auto_threshold: float
) -> UpsertStats:
    """Translate row-index groups from :func:`~.dedupe.group_edges` into path-keyed rows and upsert them."""
    items = [([records[i] for i in g.members], [(records[i].path, records[j].path, d) for i, j, d in g.edges]) for g in groups]
    return review.upsert_groups(items, policy, auto_threshold)
