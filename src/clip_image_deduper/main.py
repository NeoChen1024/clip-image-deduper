#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import os
import re
from shutil import move
from typing import List, Optional

import click
import humanize
import numpy as np
import PIL.Image
import torch
import tqdm

from .clip_encoding import CLIPImageEncoder, default_model_id
from .db_processing import load_database, update_database
from .similarity import (
    DistanceIndex,
    default_euclidean_distance_threshold,
    find_close_pairs_self,
)

keeping_modes = ["newest", "largest", "highest-quality", "pic-dir"]


def sort_highest_quality(root_dir: str, image_paths: List[str]) -> List[str]:
    # First, higher resolution images are preferred.
    # When there's JPEG and PNG versions of the same image (at same resolution), prefer PNG.
    # Then we keep the newest among the highest quality candidates.
    assert len(image_paths) > 0
    qualities = []
    for img_path in image_paths:
        full_path = os.path.join(root_dir, img_path)
        try:
            with PIL.Image.open(full_path) as img:
                width, height = img.size
                format_score = 1 if img.format == "PNG" else 0  # PNG preferred over JPEG
                qualities.append((width * height, format_score, os.path.getmtime(full_path), img_path))
        except Exception as e:
            print(f"Error evaluating image quality for {img_path}: {e}")
            qualities.append((0, 0, 0, img_path))  # Lowest quality on error
    # Sort by resolution, format, modification time
    qualities.sort(reverse=True)
    image_paths_sorted = [q[3] for q in qualities]
    return image_paths_sorted


_SOURCE_PRIORITY: list[tuple[re.Pattern, int]] = [
    (re.compile(r"[0-9]+_p[0-9]+\..*"), 4),  # Pixiv
    (re.compile(r"yande\.re [0-9]+ .*\..*"), 3),  # Yande.re
    (re.compile(r"__.*__[0-9a-f]{32}\..*"), 2),  # Danbooru
    (re.compile(r"Konachan\.com - [0-9]+ .*\..*"), 1),  # Konachan
    # others default to 0
]


def _source_score(basename: str) -> int:
    for pattern, score in _SOURCE_PRIORITY:
        if pattern.match(basename):
            return score
    return 0


def sort_image_sources(image_paths: List[str]) -> List[str]:
    return sorted(image_paths, key=lambda p: _source_score(os.path.basename(p)), reverse=True)


def is_wallpaper_dir(image_path: str) -> bool:
    dir_name = os.path.dirname(image_path)
    return "Wallpaper" in dir_name or "VWallpaper" in dir_name


def pic_dir_keeping_logic(root_dir: str, image_paths: List[str]) -> str:
    # Prefer wallpaper dirs if any exist; fall back to all paths otherwise.
    candidates = [p for p in image_paths if is_wallpaper_dir(p)] or image_paths

    # Group candidates by source score.
    by_score: dict[int, list[str]] = {}
    for p in candidates:
        s = _source_score(os.path.basename(p))
        by_score.setdefault(s, []).append(p)

    best = by_score[max(by_score)]

    if len(best) == 1:
        return best[0]

    # Multiple candidates from the same best source — use quality as tiebreaker.
    return sort_highest_quality(root_dir, best)[0]


def select_image_to_keep(root_dir: str, dup_group: List[str], keeping_logic: str) -> str:
    """Select which image to keep from a duplicate group.

    Keep tie-breakers inside a single tuple key. Chaining two ``max()`` calls
    looks reasonable, but the second call silently discards the first decision
    and makes ties depend on input order.
    """
    if keeping_logic == "newest":
        return max(dup_group, key=lambda p: (os.path.getmtime(os.path.join(root_dir, p)), os.path.getsize(os.path.join(root_dir, p))))
    if keeping_logic == "largest":
        return max(dup_group, key=lambda p: (os.path.getsize(os.path.join(root_dir, p)), os.path.getmtime(os.path.join(root_dir, p))))
    if keeping_logic == "highest-quality":
        return sort_highest_quality(root_dir, dup_group)[0]
    if keeping_logic == "pic-dir":
        return pic_dir_keeping_logic(root_dir, dup_group)
    raise ValueError(f"Unknown keeping logic: {keeping_logic}")


def find_duplicate_groups(
    image_paths: List[str],
    index: DistanceIndex,
    threshold: float,
    t,
) -> List[List[str]]:
    """Find connected duplicate groups from pairwise similarity edges.

    The dedupe relation is not guaranteed to be a clique: A may match B, B may
    match C, while A and C fall just outside the threshold. Collect all edges
    first, then merge connected components, otherwise chain duplicates get
    skipped once the middle image is marked as "already seen".

    Queries are processed in blocks of rows against the upper triangle of the
    distance matrix, so the whole N x N search is a handful of large GEMM-like
    kernels instead of N memory-bound one-vs-all passes. ``t`` is a tqdm-like
    progress object with ``update(n)`` and ``write(msg)``.
    """
    n = len(image_paths)
    parent = list(range(n))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for start, end in index.iter_blocks():
        ii, jj, dd = find_close_pairs_self(index, threshold, start, end)
        if len(ii):
            # Report per query image, like the old one-vs-all loop did.
            order = np.lexsort((jj, ii))
            ii, jj, dd = ii[order], jj[order], dd[order]
            for i in np.unique(ii):
                sel = ii == i
                matches = [(image_paths[int(j)], float(d)) for j, d in zip(jj[sel], dd[sel])]
                t.write(f"Found {len(matches)} duplicates for {image_paths[int(i)]}: {matches}")
            for i, j in zip(ii.tolist(), jj.tolist()):
                union(i, j)
        t.update(end - start)

    groups_by_root: dict[int, List[str]] = {}
    for idx, image_path in enumerate(image_paths):
        groups_by_root.setdefault(find(idx), []).append(image_path)

    return [group for group in groups_by_root.values() if len(group) > 1]


def move_duplicates(dup_group: List[str], root_dir: str, trash_dir: str, keeping_logic: str, dry_run: bool, t):
    os.makedirs(trash_dir, exist_ok=True)

    to_keep = select_image_to_keep(root_dir, dup_group, keeping_logic)

    for img_path in dup_group:
        abs_path = os.path.join(root_dir, img_path)
        try:
            if img_path != to_keep:
                dest_path = os.path.join(trash_dir, img_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                if not dry_run:
                    move(abs_path, dest_path)
                    t.write(f'Moved "{abs_path}" to trash. Keeping "{to_keep}".')
                else:
                    t.write(f'[Dry Run] Would move duplicate "{abs_path}" to trash. Keeping "{to_keep}".')
        except Exception as e:
            t.write(f'Error moving file "{abs_path}" to trash: {e}')


@click.command()
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing images to process.",
)
@click.option(
    "--db",
    "-d",
    type=click.Path(dir_okay=False),
    required=True,
    help="SQLite file storing the embedding database.",
)
@click.option(
    "--trash-dir",
    "-t",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help="Directory to move duplicate images to. If not specified, duplicates will not be moved.",
    show_default="None",
)
@click.option(
    "--clean-orphans/--no-clean-orphans",
    default=True,
    help="Whether to remove database entries whose images no longer exist.",
    show_default=True,
)
@click.option("--force-update", "-f", is_flag=True, default=False, help="Force update all images, ignoring modification times.")
@click.option(
    "--device",
    "-c",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to run the CLIP model on.",
    show_default=True,
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=4,
    help="Batch size for processing images when updating the database.",
    show_default=True,
)
@click.option("--model-id", "-m", default=default_model_id, help="CLIP model identifier.", show_default=True)
@click.option("--skip-update", is_flag=True, default=False, help="Skip the database update step.")
@click.option(
    "--dry-run",
    "-n",
    is_flag=True,
    default=False,
    help="Preview duplicate moves without moving files. Database files are still refreshed unless --skip-update is set.",
)
@click.option(
    "--threshold",
    "-th",
    type=float,
    default=default_euclidean_distance_threshold,
    help="Euclidean distance threshold for considering images as duplicates.",
    show_default=True,
)
@click.option(
    "--keeping-logic",
    "-kl",
    type=click.Choice(keeping_modes, case_sensitive=False),
    default="largest",
    help="Which copy to keep among duplicates.",
    show_default=True,
)
def main(
    image_dir: str,
    db: str,
    model_id: str,
    force_update: bool,
    clean_orphans: bool,
    device: str,
    skip_update: bool,
    dry_run: bool,
    threshold: float,
    trash_dir: str,
    keeping_logic: str,
    batch_size: int = 4,
):
    torch.set_float32_matmul_precision("highest")  # use highest precision for best accuracy in distance calculations
    if not skip_update:
        # Dry-run is about avoiding file moves; skipping DB refresh would make
        # the preview stale. Use --skip-update when a no-write preview matters
        # more than accuracy.
        print("Updating database...")
        encoder = CLIPImageEncoder(model_id=model_id, device=device)
        update_database(encoder, image_dir, db, force_update, clean_orphans, batch_size=batch_size)
        encoder.cleanup()
        del encoder
        gc.collect()

    print("Loading database...")
    image_paths, embeddings_db = load_database(db)  # (N, D)
    print(f"Loaded {len(image_paths)} entries in the database.")
    if len(image_paths) == 0:
        print("No entries found in the database. Exiting.")
        raise SystemExit(1)

    index = DistanceIndex(embeddings_db, device)
    del embeddings_db
    print(
        f"Embeddings shape: ({index.n}, {index.dim}), device memory: {humanize.naturalsize(index.nbytes, binary=True)}, "
        f"backend: {index.backend_name()}"
    )

    print("Finding duplicates...")
    t = tqdm.tqdm(total=len(image_paths), desc="Processing images", unit="image")
    duplicate_groups = find_duplicate_groups(image_paths, index, threshold, t)
    t.close()
    duplicate_image_count = sum(len(group) - 1 for group in duplicate_groups)

    if trash_dir is not None:
        for dup_group in duplicate_groups:
            move_duplicates(dup_group, image_dir, trash_dir, keeping_logic, dry_run, t)

    dry_run_str = ""
    if dry_run:
        dry_run_str = " (dry run, no files were moved)"

    print(
        f"Deduplication complete{dry_run_str}, processed {len(image_paths)} images, "
        f"found {duplicate_image_count} duplicates across {len(duplicate_groups)} groups."
    )

    index.release()
    del index
    gc.collect()
    torch.cuda.empty_cache()
    torch.compiler.reset()
