#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This module is for updating the "database" of clip_image_deduper. It scans a specified directory for valid images,
# If an image's modification time is same or newer than the existing .npz file, it calls a provided function to process the image.

import json
import math
import os
from typing import Any, Callable, Dict, List, Tuple

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


def update_database_generic(
    image_dir: str,
    db_dir: str,
    process_image_func: Callable,
    data_extension: str,
    force_update: bool = False,
    clean_orphans: bool = True,
):
    """Update the database of images by processing each image in the specified directory."""
    t = tqdm.tqdm(list(walk_directory_relative(image_dir)))
    for relative_path in t:
        try:
            image_path = os.path.join(image_dir, relative_path)
            data_path = os.path.join(db_dir, f"{relative_path}{data_extension}")

            image_mtime = os.path.getmtime(image_path)
            if os.path.exists(data_path) and not force_update:
                db_mtime = os.path.getmtime(data_path)
            else:
                db_mtime = -math.inf
                # try to create parent directories
                os.makedirs(os.path.dirname(data_path), exist_ok=True)

            if image_mtime < db_mtime:
                t.write(f"Skipping up-to-date image: {image_path}")
                continue

            if not verify_image(image_path):
                t.write(f"Skipping invalid image: {image_path}")
                continue

            t.write(f"Processing image: {image_path}")
            process_image_func(image_path, data_path)
        except Exception as e:
            t.write(f"Error processing image {relative_path}: {e}")

    if clean_orphans:
        # walk through db_dir to find and remove orphaned data files
        t = tqdm.tqdm(list(walk_directory_relative(db_dir)))
        t.write("Checking for orphaned data files...")
        for relative_path in t:
            try:
                data_path = os.path.join(db_dir, relative_path)
                image_path = os.path.join(image_dir, relative_path[: -len(data_extension)])

                if not os.path.exists(image_path):
                    t.write(f"Removing orphaned data file: {data_path}")
                    os.remove(data_path)
            except Exception as e:
                t.write(f"Error checking data file {relative_path}: {e}")


def load_database_generic(db_dir: str, data_extension: str, load_data_func: Callable) -> Tuple[List[str], List[Any]]:
    """Load all database files from the db directory."""
    file_paths = []
    db_data = []
    t = tqdm.tqdm(list(walk_directory_relative(db_dir)))
    for relative_path in t:
        if relative_path.endswith(data_extension):
            db_file_path = os.path.join(db_dir, relative_path)
            try:
                data = load_data_func(db_file_path)
                file_paths.append(relative_path.removesuffix(data_extension))
                db_data.append(data)
            except Exception as e:
                t.write(f"Error loading database file {db_file_path}: {e}")
    return file_paths, db_data


def update_db(image_dir: str, db_dir: str, force_update: bool, clean_orphans: bool, clip_model: str, device: str):
    print(f"Using CLIP model: {clip_model}, on device {device}")
    encoder = CLIPImageEncoder(model_id=clip_model, device=device)

    # don't need to catch exceptions here, they are handled in update_database_generic
    def process_image(image_path: str, db_npz_path: str):
        with PIL.Image.open(image_path) as img:
            img = img.convert("RGB")
            encoding = encoder.encode_image(img)

        # save embedding as NumPy npz file
        np.savez_compressed(db_npz_path, clip_embedding=encoding)

    update_database_generic(
        image_dir, db_dir, process_image, data_extension=".npz", force_update=force_update, clean_orphans=clean_orphans
    )


def load_db(db_dir: str) -> Tuple[List[str], List[ndarray]]:
    def load_npz(f) -> ndarray:
        data = np.load(f)
        return data["clip_embedding"]

    filenames, database = load_database_generic(db_dir, data_extension=".npz", load_data_func=load_npz)
    return filenames, database  # could be huge


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
    help="Directory to store the database JSON files.",
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
def main(image_dir: str, db_dir: str, force_update: bool, clean_orphans: bool, clip_model: str, device: str):
    print("Starting database update...")
    update_db(image_dir, db_dir, force_update, clean_orphans, clip_model, device)
    print("Try loading the database...")
    fn, db = load_db(db_dir)
    print(f"Loaded {len(db)} entries in the database.")


if __name__ == "__main__":
    main()
