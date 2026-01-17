#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Remove duplicate images from a "import directory" by moving them to a trash directory.

from email.mime import base
import os
import gc
from shutil import move
from typing import List, Optional, Tuple, Any

import click
import humanize
import numpy as np
import torch
import tqdm

from .clip_encoding import CLIPImageEncoder, default_model_id
from .db_processing import load_database, update_database
from .similarity import (
    default_euclidean_distance_threshold,
    find_similar_images_euclidean,
)


def move_duplicate(image_path: str, root_dir: str, trash_dir: str, dry_run: bool, t):
    os.makedirs(trash_dir, exist_ok=True)

    abs_path = os.path.join(root_dir, image_path)
    try:
        dest_path = os.path.join(trash_dir, image_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        if not dry_run:
            move(abs_path, dest_path)
            t.write(f'Moved "{abs_path}" to trash.')
        else:
            t.write(f'[Dry Run] Would move duplicate "{abs_path}" to trash.')
    except Exception as e:
        t.write(f'Error moving file "{abs_path}" to trash: {e}')


@click.command()
@click.option(
    "--base-image-dir",
    "-bi",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing base images to process.",
)
@click.option(
    "--base-db-dir",
    "-bd",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Directory to store the base database files.",
)
@click.option(
    "--import-image-dir",
    "-ii",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing import images to process.",
)
@click.option(
    "--import-db-dir",
    "-id",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Directory to store the import database files.",
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
def main(
    base_image_dir: str,
    base_db_dir: str,
    import_image_dir: str,
    import_db_dir: str,
    model_id: str,
    force_update: bool,
    clean_orphans: bool,
    device: str,
    skip_update: bool,
    dry_run: bool,
    threshold: float,
    trash_dir: str,
):
    if not skip_update and not dry_run:
        encoder = CLIPImageEncoder(model_id=model_id, device=device)
        print("Updating base database...")
        update_database(encoder, base_image_dir, base_db_dir, force_update, clean_orphans)
        print("Updating import database...")
        update_database(encoder, import_image_dir, import_db_dir, force_update, clean_orphans)
        gc.collect()

    def db_processing(db_type: str, db_dir: str) -> Tuple[List[str], np.ndarray, torch.Tensor]:
        print(f"Loading {db_type} database...")
        image_paths, database = load_database(db_dir)
        print(f"Loaded {len(database)} entries in the database.")
        if len(database) == 0:
            print(f"No entries found in the {db_type} database. Exiting.")
            raise SystemExit(1)

        # put all image paths and embeddings into lists for easier processing
        print(f"Preparing {db_type} embeddings...")
        embeddings_db = np.stack(database, axis=0)  # (N, D)
        del database
        gc.collect()
        print(
            f"Embeddings DB shape of {db_type}: {embeddings_db.shape}, memory size: {humanize.naturalsize(embeddings_db.nbytes, binary=True)}"
        )
        embeddings_torch = torch.from_numpy(embeddings_db).to(device).float()

        return image_paths, embeddings_db, embeddings_torch

    base_image_paths, base_embeddings_db, base_embeddings_torch = db_processing("base", base_db_dir)
    import_image_paths, import_embeddings_db, import_embeddings_torch = db_processing("import", import_db_dir)

    print("Finding duplicates...")
    duplicate_count = 0
    t = tqdm.tqdm(import_image_paths, desc="Processing import images", unit="image")

    for idx, image_path in enumerate(t):
        image_embedding = import_embeddings_db[idx]  # shape (D)

        similar_images = find_similar_images_euclidean(-1, image_embedding, base_embeddings_torch, threshold=threshold)
        if similar_images:
            similar_images_paths = [(base_image_paths[s_idx], sim) for s_idx, sim in similar_images]
            t.write(f"Found {len(similar_images)} duplicates for {image_path}: {similar_images_paths}")
            duplicate_count += len(similar_images)
            if trash_dir is not None:
                move_duplicate(image_path, import_image_dir, trash_dir, dry_run, t)

    dry_run_str = ""
    if dry_run:
        dry_run_str = " (dry run, no files were moved)"

    print(f"Dedupelication complete., processed {len(import_image_paths)} images, found {duplicate_count} duplicates.{dry_run_str}")
