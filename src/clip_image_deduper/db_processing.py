#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This module is for updating the "database" of clip_image_deduper. It scans a specified directory for valid images,
# If an image's modification time is same or newer than the existing .npz file, it calls a provided function to process the image.

import math
import os
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import numpy as np
import PIL.Image
import torch
import tqdm
from numpy import ndarray

from .clip_encoding import CLIPImageEncoder, default_model_id


def walk_directory_relative(directory: str):
    """Walk through a directory and yield relative file paths."""
    for root, _, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, directory)
            yield relative_path


def verify_image(image_path: str) -> bool:
    """Verify if an image can be opened."""
    try:
        with PIL.Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def update_database(
    encoder: CLIPImageEncoder,
    image_dir: str,
    db_dir: str,
    force_update: bool = False,
    clean_orphans: bool = True,
):
    """Update the database of images by processing each image in the specified directory."""
    t = tqdm.tqdm(list(walk_directory_relative(image_dir)))
    for relative_path in t:
        try:
            image_path = os.path.join(image_dir, relative_path)
            data_path = os.path.join(db_dir, f"{relative_path}.npz")

            image_mtime = os.path.getmtime(image_path)
            if os.path.exists(data_path) and not force_update:
                db_mtime = os.path.getmtime(data_path)
            else:
                db_mtime = -math.inf
                # try to create parent directories
                os.makedirs(os.path.dirname(data_path), exist_ok=True)

            if image_mtime < db_mtime:
                continue

            if not verify_image(image_path):
                t.write(f"Skipping invalid image: {image_path}")
                continue

            t.write(f"Processing image: {image_path}")
            with PIL.Image.open(image_path) as img:
                img = img.convert("RGB")
                encoding = encoder.encode_image(img)

            # save embedding as NumPy npz file
            np.savez_compressed(data_path, clip_embedding=encoding)
        except Exception as e:
            t.write(f"Error processing image {relative_path}: {e}")

    if clean_orphans:
        # walk through db_dir to find and remove orphaned data files
        t = tqdm.tqdm(list(walk_directory_relative(db_dir)))
        t.write("Checking for orphaned data files...")
        for relative_path in t:
            try:
                data_path = os.path.join(db_dir, relative_path)
                image_path = os.path.join(image_dir, relative_path.removesuffix(".npz"))  # remove .npz extension

                if not os.path.exists(image_path):
                    t.write(f"Removing orphaned data file: {data_path}")
                    os.remove(data_path)
            except Exception as e:
                t.write(f"Error checking data file {relative_path}: {e}")


def _load_single_db_file(args: Tuple[str, str]) -> Optional[Union[Tuple[str, ndarray], Tuple[str, str]]]:
    """Helper function to load a single database file.

    This is defined at module level so it can be used with multiprocessing.Pool.
    """
    db_dir, relative_path = args

    if not relative_path.endswith(".npz"):
        return None

    db_file_path = os.path.join(db_dir, relative_path)
    try:
        data = np.load(db_file_path)
        return relative_path.removesuffix(".npz"), data["clip_embedding"]
    except Exception as e:
        # Return an error marker and message so the caller can log it.
        return "", f"Error loading database file {db_file_path}: {e}"


def load_database(db_dir: str) -> Tuple[List[str], List[np.ndarray]]:
    """Load all database files from the db directory."""
    file_paths: List[str] = []
    db_data: List[np.ndarray] = []

    file_list = list(walk_directory_relative(db_dir))
    npz_files = [p for p in file_list if p.endswith(".npz")]

    if not npz_files:
        return file_paths, db_data

    t = tqdm.tqdm(total=len(npz_files))

    # Use a process pool to load database files in parallel.
    with mp.Pool() as pool:
        for result in pool.imap_unordered(_load_single_db_file, ((db_dir, p) for p in npz_files)):
            if result is None:
                # Non-npz files are filtered out in the worker, but keep for safety.
                continue

            rel_path, data = result
            if type(data) is str:
                # Log errors via tqdm to keep output consistent with the rest of the module.
                t.write(data)
            elif isinstance(data, np.ndarray):
                file_paths.append(rel_path)
                db_data.append(data)

            t.update(1)
    return file_paths, db_data


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
@click.option("--clip-model", "-m", default=default_model_id, help="CLIP model to use for encoding images.", show_default=True)
@click.option(
    "--device",
    "-c",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to run the CLIP model on.",
    show_default=True,
)
@click.option(
    "--skip-update",
    is_flag=True,
    default=False,
    help="Skip updating the database and only load existing data.",
)
def main(image_dir: str, db_dir: str, force_update: bool, clean_orphans: bool, clip_model: str, device: str, skip_update: bool):
    if not skip_update:
        print("Starting database update...")
        print(f"Using CLIP model: {clip_model} on device: {device}")
        encoder = CLIPImageEncoder(model_id=clip_model, device=device)
        update_database(encoder, image_dir, db_dir, force_update, clean_orphans)
    print("Try loading the database...")
    fn, db = load_database(db_dir)
    print(f"Loaded {len(db)} entries in the database.")


if __name__ == "__main__":
    main()
