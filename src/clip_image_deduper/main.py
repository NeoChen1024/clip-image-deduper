#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
from shutil import move
from typing import List, Optional

import click
import humanize
import numpy as np
import torch
import tqdm
import re

import PIL.Image

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


# dir structure: Anime, Wallpaper, VWallpaper
# image sources according to preference (high to low): Pixiv (illust_id_pX.*), Yande.re (yande.re Y_ID *.*)
#   Danbooru (__*__MD5.*), and Konachan (Konachan.com - K_ID *.*), then others (e.g. Twitter/X or misc image sources)
match_pixiv = re.compile(r"[0-9]+_p[0-9]+\..*")
match_yande_re = re.compile(r"yande\.re [0-9]+ .*\..*")
match_danbooru = re.compile(r"__.*__[0-9a-f]{32}\..*")
match_konachan = re.compile(r"Konachan\.com - [0-9]+ .*\..*")


def sort_image_sources(image_paths: List[str]) -> List[str]:
    image_path_scores = []  # Tuple[int, str]
    for image_path in image_paths:
        img_basename = os.path.basename(image_path)
        score = 0
        if match_pixiv.match(img_basename):
            score += 4
        elif match_yande_re.match(img_basename):
            score += 3
        elif match_danbooru.match(img_basename):
            score += 2
        elif match_konachan.match(img_basename):
            score += 1
        else:
            score += 0
        image_path_scores.append((score, image_path))
    # Sort by score descending
    image_paths_sorted = sorted(image_path_scores, key=lambda x: x[0], reverse=True)
    return [p[1] for p in image_paths_sorted]


def is_wallpaper_dir(image_path: str) -> bool:
    dir_name = os.path.dirname(image_path)
    return "Wallpaper" in dir_name or "VWallpaper" in dir_name


def pic_dir_keeping_logic(root_dir: str, image_paths: List[str]) -> str:
    # First, prefer to keep images in Wallpaper and VWallpaper:
    wallpaper_images = [p for p in image_paths if is_wallpaper_dir(p)]  # image_path is relative path
    if len(wallpaper_images) > 0:
        # there's same images in wallpapers and Anime dir, keep the copies in wallpaper.
        sources = sort_image_sources(wallpaper_images)
        hq = sort_highest_quality(root_dir, wallpaper_images)
        return sources[0]

    # Else, keep from Anime or other dirs
    sources = sort_image_sources(image_paths)
    hq = sort_highest_quality(root_dir, image_paths)
    return sources[0]


def move_duplicates(dup_group: List[str], root_dir: str, trash_dir: str, keeping_logic: str, dry_run: bool, t):
    os.makedirs(trash_dir, exist_ok=True)

    # Determine which image to keep based on the keeping logic
    if keeping_logic == "newest":
        # add size to break ties
        to_keep = max(dup_group, key=lambda p: os.path.getsize(os.path.join(root_dir, p)))
        to_keep = max(dup_group, key=lambda p: os.path.getmtime(os.path.join(root_dir, p)))
    elif keeping_logic == "largest":
        # add mtime to break ties
        to_keep = max(dup_group, key=lambda p: os.path.getmtime(os.path.join(root_dir, p)))
        to_keep = max(dup_group, key=lambda p: os.path.getsize(os.path.join(root_dir, p)))
    elif keeping_logic == "highest-quality":
        to_keep = sort_highest_quality(root_dir, dup_group)[0]
    elif keeping_logic == "pic-dir":
        # Keeping logic based on directory structure and image source
        to_keep = pic_dir_keeping_logic(root_dir, dup_group)
    else:
        raise ValueError(f"Unknown keeping logic: {keeping_logic}")

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
@click.option("--model-id", "-m", default=default_model_id, help="CLIP model identifier.", show_default=True)
@click.option("--skip-update", is_flag=True, default=False, help="Skip the database update step.")
@click.option("--dry-run", "-n", is_flag=True, default=False, help="Perform a dry run without making any changes.")
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
):
    torch.set_float32_matmul_precision("highest")  # use highest precision for best accuracy in distance calculations
    if not skip_update and not dry_run:
        print("Updating database...")
        encoder = CLIPImageEncoder(model_id=model_id, device=device)
        update_database(encoder, image_dir, db_dir, force_update, clean_orphans)

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
    duplicates = {}  # dict: {"image_path": bool} to mark images already recorded as duplicates
    t = tqdm.tqdm(image_paths, desc="Processing images", unit="image")

    for idx, image_path in enumerate(t):
        image_embedding = embeddings_db[idx]  # shape (D)

        # Only compare this image against images that come after it in the
        # list to avoid checking each pair twice (i,j) and (j,i).
        # (upper-triangular comparison)
        database_slice_torch = embeddings_torch[idx + 1 :]
        if database_slice_torch.size(0) == 0:
            continue

        similar_images = find_similar_images_euclidean(idx, image_embedding, database_slice_torch, threshold=threshold)
        if similar_images:
            # Adjust indices from slice-local [0, ...) back to global indices.
            similar_images = [(s_idx + idx + 1, sim) for s_idx, sim in similar_images]
            # check if this image is already recorded in duplicates (can happen when there's more than 2 duplicates)
            if image_path in duplicates:
                return
            current_dupes = [image_path] + [image_paths[s_idx] for s_idx, _ in similar_images]
            for img_path in current_dupes:
                duplicates[img_path] = True
            similar_images_paths = [(image_paths[s_idx], sim) for s_idx, sim in similar_images]
            t.write(f"Found {len(similar_images)} duplicates for {image_path}: {similar_images_paths}")
            if trash_dir is not None:
                move_duplicates(current_dupes, image_dir, trash_dir, keeping_logic, dry_run, t)

    dry_run_str = ""
    if dry_run:
        dry_run_str = " (dry run, no files were moved)"

    print(f"Dedupelication complete{dry_run_str}, processed {len(image_paths)} images, found {len(duplicates)} duplicates.")
