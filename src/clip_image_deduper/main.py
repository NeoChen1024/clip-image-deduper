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
    default_euclidean_distance_threshold,
    find_similar_images_euclidean,
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
    embeddings_db: np.ndarray,
    embeddings_torch: torch.Tensor,
    threshold: float,
    t,
) -> List[List[str]]:
    """Find connected duplicate groups from pairwise similarity edges.

    The dedupe relation is not guaranteed to be a clique: A may match B, B may
    match C, while A and C fall just outside the threshold. Collect all edges
    first, then merge connected components, otherwise chain duplicates get
    skipped once the middle image is marked as "already seen".
    """
    parent = list(range(len(image_paths)))

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

    for idx, image_path in enumerate(t):
        image_embedding = embeddings_db[idx]  # shape (D)

        # Search only the upper-triangular slice to avoid duplicate work. The
        # query image is not inside this slice, so self-filtering must be
        # disabled with image_idx=-1.
        database_slice_torch = embeddings_torch[idx + 1 :]
        if database_slice_torch.size(0) == 0:
            continue

        similar_images = find_similar_images_euclidean(-1, image_embedding, database_slice_torch, threshold=threshold)
        if not similar_images:
            continue

        similar_images = [(s_idx + idx + 1, sim) for s_idx, sim in similar_images]
        similar_images_paths = [(image_paths[s_idx], sim) for s_idx, sim in similar_images]
        t.write(f"Found {len(similar_images)} duplicates for {image_path}: {similar_images_paths}")
        for s_idx, _ in similar_images:
            union(idx, s_idx)

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
    "--db-dir",
    "-d",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Directory to store the database files.",
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
    help="Whether to remove orphaned database files that no longer have corresponding images.",
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
    db_dir: str,
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
        update_database(encoder, image_dir, db_dir, force_update, clean_orphans, batch_size=batch_size)
        encoder.cleanup()
        del encoder
        gc.collect()

    print("Loading database...")
    image_paths, database = load_database(db_dir)
    print(f"Loaded {len(database)} entries in the database.")
    if len(database) == 0:
        print("No entries found in the database. Exiting.")
        raise SystemExit(1)

    # put all image paths and embeddings into lists for easier processing
    print("Preparing embeddings...")
    embeddings_db = np.stack(database, axis=0)  # (N, D)
    del database
    gc.collect()
    print(f"Embeddings shape: {embeddings_db.shape}, memory size: {humanize.naturalsize(embeddings_db.nbytes, binary=True)}")
    embeddings_torch = torch.from_numpy(embeddings_db).to(device).float()

    print("Finding duplicates...")
    t = tqdm.tqdm(image_paths, desc="Processing images", unit="image")
    duplicate_groups = find_duplicate_groups(image_paths, embeddings_db, embeddings_torch, threshold, t)
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

    del embeddings_torch
    del embeddings_db
    gc.collect()
    torch.cuda.empty_cache()
    torch.compiler.reset()
