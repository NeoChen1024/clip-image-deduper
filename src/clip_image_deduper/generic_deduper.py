#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import click
import humanize
import numpy as np
import torch
import tqdm

from .clip_encoding import cosine_similarity, default_model_id
from .db_processing import load_db, update_db, verify_image, walk_directory_relative


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
def main(
    image_dir: str,
    db_dir: str,
    model_id: str,
    force_update: bool,
    clean_orphans: bool,
    device: str,
    skip_update: bool,
):
    if not skip_update:
        update_db(image_dir, db_dir, force_update, clean_orphans, model_id, device)
    print("Loading database...")
    image_paths, database = load_db(db_dir)
    print(f"Loaded {len(database)} entries in the database.")

    # put all image paths and embeddings into lists for easier processing
    print("Preparing embeddings...")
    embeddings = np.stack(database, axis=0)  # shape (N, D)
    print(f"Embeddings shape: {embeddings.shape}, memory size: {humanize.naturalsize(embeddings.nbytes, binary=True)}")
