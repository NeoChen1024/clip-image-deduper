#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This module is for updating the "database" of clip_image_deduper. It scans a specified directory for valid images,
# If an image's modification time is same or newer than the existing .npz file, it calls a provided function to process the image.

import math
import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _load_and_prepare_image(image_dir: str, relative_path: str) -> Tuple[str, Union[PIL.Image.Image, Exception]]:
    """Load and validate a single image, returning either a PIL image or an Exception.

    This is intended to be used with ThreadPoolExecutor for asynchronous IO.
    """
    image_path = os.path.join(image_dir, relative_path)

    try:
        # Reuse existing validation logic.
        if not verify_image(image_path):
            raise ValueError("Invalid image")

        # Load and convert to RGB; ensure data is fully loaded into memory.
        with PIL.Image.open(image_path) as img:
            img = img.convert("RGB")
            img.load()

        return relative_path, img
    except Exception as e:
        return relative_path, e


def _encode_and_save_batch(
    encoder: CLIPImageEncoder,
    db_dir: str,
    batch_paths: List[str],
    batch_images: List[PIL.Image.Image],
) -> None:
    """Encode a batch of images and save their embeddings to disk."""
    if not batch_images:
        return

    embeddings = encoder.encode_images(batch_images)
    for rel_path, embedding in zip(batch_paths, embeddings):
        data_path = os.path.join(db_dir, f"{rel_path}.npz")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        np.savez_compressed(data_path, clip_embedding=embedding)


def update_database(
    encoder: CLIPImageEncoder,
    image_dir: str,
    db_dir: str,
    force_update: bool = False,
    clean_orphans: bool = True,
    batch_size: int = 4,
):
    """Update the database of images by processing each image in the specified directory."""
    # First determine which images actually need to be (re)encoded based on mtime.
    candidates: List[str] = []
    for relative_path in walk_directory_relative(image_dir):
        try:
            image_path = os.path.join(image_dir, relative_path)
            data_path = os.path.join(db_dir, f"{relative_path}.npz")

            image_mtime = os.path.getmtime(image_path)
            if os.path.exists(data_path) and not force_update:
                db_mtime = os.path.getmtime(data_path)
            else:
                db_mtime = -math.inf

            if image_mtime < db_mtime:
                continue

            candidates.append(relative_path)
        except Exception:
            # Ignore pathological filesystem issues here; they will be surfaced later if needed.
            continue

    # Asynchronously load and validate candidate images, then encode them in batches.
    if candidates:
        t = tqdm.tqdm(total=len(candidates))
        batch_paths: List[str] = []
        batch_images: List[PIL.Image.Image] = []

        # Limit the maximum number of in-flight futures to a small
        # multiple of the encoding batch size to avoid holding an
        # unbounded number of Future objects in memory.
        max_in_flight_futures = max(batch_size * 2, 1)

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            num_candidates = len(candidates)
            for start in range(0, num_candidates, max_in_flight_futures):
                chunk = candidates[start : start + max_in_flight_futures]
                futures = [executor.submit(_load_and_prepare_image, image_dir, p) for p in chunk]

                for future in as_completed(futures):
                    rel_path, result = future.result()

                    if isinstance(result, Exception):
                        image_path = os.path.join(image_dir, rel_path)
                        t.write(f"Skipping invalid image: {image_path} ({result})")
                    else:
                        batch_paths.append(rel_path)
                        batch_images.append(result)

                        if len(batch_images) >= batch_size:
                            _encode_and_save_batch(encoder, db_dir, batch_paths, batch_images)
                            batch_paths = []
                            batch_images = []

                    t.update(1)

        # Flush any remaining images.
        if batch_images:
            _encode_and_save_batch(encoder, db_dir, batch_paths, batch_images)

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
    "--batch-size",
    "-b",
    type=int,
    default=4,
    help="Batch size for processing images when updating the database.",
    show_default=True,
)
@click.option(
    "--skip-update",
    is_flag=True,
    default=False,
    help="Skip updating the database and only load existing data.",
)
def main(
    image_dir: str,
    db_dir: str,
    force_update: bool,
    clean_orphans: bool,
    clip_model: str,
    device: str,
    skip_update: bool,
    batch_size: int,
):
    if not skip_update:
        print("Starting database update...")
        encoder = CLIPImageEncoder(model_id=clip_model, device=device)
        update_database(encoder, image_dir, db_dir, force_update, clean_orphans, batch_size=batch_size)
    print("Try loading the database...")
    fn, db = load_database(db_dir)
    print(f"Loaded {len(db)} entries in the database.")


if __name__ == "__main__":
    main()
