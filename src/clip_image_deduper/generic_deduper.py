#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import click
import humanize
import numpy as np
import torch
import tqdm

from .clip_encoding import default_model_id
from .db_processing import load_db, update_db, verify_image, walk_directory_relative
from .similarity import (
    default_cosine_similarity_threshold,
    default_euclidean_distance_threshold,
    find_similar_images_cosine,
    find_similar_images_euclidean,
)


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
    "--cosine-similarity-threshold",
    "-ct",
    type=float,
    default=default_cosine_similarity_threshold,
    help="Cosine similarity threshold for considering images as duplicates.",
    show_default=True,
)
@click.option(
    "--euclidean-distance-threshold",
    "-et",
    type=float,
    default=default_euclidean_distance_threshold,
    help="Euclidean distance threshold for considering images as duplicates.",
    show_default=True,
)
@click.option(
    "--detection-method",
    "-dm",
    type=click.Choice(["cosine", "euclidean"], case_sensitive=False),
    default="cosine",
    help="Method to use for duplicate detection.",
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
    cosine_similarity_threshold: float,
    euclidean_distance_threshold: float,
    detection_method: str,
):
    if not skip_update and not dry_run:
        update_db(image_dir, db_dir, force_update, clean_orphans, model_id, device)
    print("Loading database...")
    image_paths, database = load_db(db_dir)
    print(f"Loaded {len(database)} entries in the database.")

    # put all image paths and embeddings into lists for easier processing
    print("Preparing embeddings...")
    embeddings_db = np.stack(database, axis=0)  # shape (N, D)
    print(f"Embeddings shape: {embeddings_db.shape}, memory size: {humanize.naturalsize(embeddings_db.nbytes, binary=True)}")

    print("Finding duplicates...")
    duplicates = []  # list of [image_path, dupe_image_path, dupe_image_path2, ...]
    t = tqdm.tqdm(image_paths, desc="Processing images", unit="image")

    def process_duplicate(method, image_path, similar_images, t):
        # check if this image is already recorded in duplicates (can happen when there's more than 2 duplicates)
        already_present = False
        for dup_group in duplicates:
            if image_path in dup_group:
                already_present = True
                break
        if already_present:
            return

        duplicates.append([image_path] + [image_paths[s_idx] for s_idx, _ in similar_images])
        similar_images_paths = [(image_paths[s_idx], sim) for s_idx, sim in similar_images]
        t.write(f"{method}: {len(similar_images)} duplicates for {image_path}: {similar_images_paths}")

    # TODO: parallelize this loop
    for idx, image_path in enumerate(t):
        image_embedding = embeddings_db[idx]  # shape (D)

        # Only compare this image against images that come after it in the
        # list to avoid checking each pair twice (i,j) and (j,i).
        # (upper-triangular comparison)
        database_slice = embeddings_db[idx + 1 :]
        if database_slice.size == 0:
            continue

        similar_images = []
        if detection_method == "cosine":
            similar_images = find_similar_images_cosine(idx, image_embedding, database_slice, threshold=cosine_similarity_threshold)
        elif detection_method == "euclidean":
            similar_images = find_similar_images_euclidean(
                idx, image_embedding, database_slice, threshold=euclidean_distance_threshold
            )
        if similar_images:
            # Adjust indices from slice-local [0, ...) back to global indices.
            similar_images = [(s_idx + idx + 1, sim) for s_idx, sim in similar_images]
            process_duplicate(detection_method, image_path, similar_images, t)

    if dry_run:
        print("Dry run complete. No changes were made.")
        return
