"""Duplicate detection on top of :class:`~.similarity.DistanceIndex`, and moving losers to a trash directory."""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable, Sequence

import numpy as np

from .db_store import ImageRecord
from .keeping import Policy
from .similarity import DistanceIndex, find_close_pairs_cross, find_close_pairs_self

logger = logging.getLogger(__name__)

Progress = Callable[[int], object] | None


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> list[list[int]]:
        by_root: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            by_root.setdefault(self.find(i), []).append(i)
        return [g for g in by_root.values() if len(g) > 1]


def _log_matches(prefix: str, query_paths: Sequence[str], match_paths: Sequence[str], ii: np.ndarray, jj: np.ndarray, dd: np.ndarray) -> None:
    order = np.lexsort((jj, ii))
    ii, jj, dd = ii[order], jj[order], dd[order]
    for i in np.unique(ii):
        sel = ii == i
        matches = [(match_paths[int(j)], round(float(d), 4)) for j, d in zip(jj[sel], dd[sel])]
        logger.info("%s %s: %s", prefix, query_paths[int(i)], matches)


def find_duplicate_groups(paths: Sequence[str], index: DistanceIndex, threshold: float, progress: Progress = None) -> list[list[int]]:
    """Connected components of the "distance <= threshold" relation, as lists of row indices.

    The relation is not transitive: A may match B and B match C while A and C fall just outside the threshold, so all
    edges are collected first and merged with union-find. Only the upper triangle is searched, in blocks of rows.
    """
    uf = _UnionFind(index.n)
    for start, end in index.iter_blocks():
        ii, jj, dd = find_close_pairs_self(index, threshold, start, end)
        if len(ii):
            _log_matches("Duplicates of", paths, paths, ii, jj, dd)
            for i, j in zip(ii.tolist(), jj.tolist()):
                uf.union(i, j)
        if progress:
            progress(end - start)
    return uf.groups()


def find_cross_duplicates(
    query_paths: Sequence[str], queries: DistanceIndex, base_paths: Sequence[str], base: DistanceIndex, threshold: float, progress: Progress = None
) -> dict[int, int]:
    """For every query row that has a match in ``base``: ``{query_row: number_of_matches}``."""
    hits: dict[int, int] = {}
    for start, end in queries.iter_blocks(other=base):
        ii, jj, dd = find_close_pairs_cross(queries, base, threshold, start, end)
        if len(ii):
            _log_matches("Already in base:", query_paths, base_paths, ii, jj, dd)
            for i, count in zip(*np.unique(ii, return_counts=True)):
                hits[int(i)] = int(count)
        if progress:
            progress(end - start)
    return hits


def move_to_trash(root_dir: str, relative_path: str, trash_dir: str, *, dry_run: bool, reason: str = "") -> bool:
    """Move one file into ``trash_dir`` preserving its relative path. Returns True when moved (or would be, in dry run)."""
    src = os.path.join(root_dir, relative_path)
    dst = os.path.join(trash_dir, relative_path)
    suffix = f" ({reason})" if reason else ""
    if dry_run:
        logger.info('[dry run] would move "%s" to trash%s', src, suffix)
        return True
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
    except OSError as e:
        logger.error('Error moving "%s" to trash: %s', src, e)
        return False
    logger.info('Moved "%s" to trash%s', src, suffix)
    return True


def trash_duplicate_groups(
    groups: Sequence[Sequence[int]], records: Sequence[ImageRecord], root_dir: str, trash_dir: str, policy: Policy, *, dry_run: bool
) -> int:
    """Apply ``policy`` to each group, keep the winner, move the rest. Returns the number of files moved."""
    moved = 0
    for group in groups:
        candidates = [records[i] for i in group]
        keep = policy.select(candidates)
        for rec in candidates:
            if rec is not keep and move_to_trash(root_dir, rec.path, trash_dir, dry_run=dry_run, reason=f"keeping {keep.path}"):
                moved += 1
    return moved
