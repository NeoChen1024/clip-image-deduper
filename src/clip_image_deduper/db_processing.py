#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This module is for updating the "database" of clip_image_deduper. It scans a specified directory for valid images,
# If an image's modification time is same or newer than the existing .json file, it calls a provided function to process the image.

import math
import os
import PIL.Image
from typing import List, Callable, Dict
import tqdm
import click
import json


def walk_directory_relative(directory: str):
    """Walk through a directory and yield relative file paths."""
    for root, _, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, directory)
            yield relative_path


def get_image_modification_time(image_path: str) -> float:
    """Get the modification time of an image file."""
    return os.path.getmtime(image_path)


def verify_image(image_path: str) -> bool:
    """Verify if an image can be opened."""
    try:
        with PIL.Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def update_database(image_dir: str, db_dir: str, process_image_func: Callable, data_extension: str, force_update: bool = False):
    """Update the database of images by processing each image in the specified directory."""
    t = tqdm.tqdm(list(walk_directory_relative(image_dir)))
    for relative_path in t:
        try:
            image_path = os.path.join(image_dir, relative_path)
            data_path = os.path.join(db_dir, f"{relative_path}{data_extension}")

            image_mtime = get_image_modification_time(image_path)
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


def load_database(db_dir: str) -> Dict[str, dict]:
    """Load all JSON database files from the db directory."""
    database = {}
    relative_paths = walk_directory_relative(db_dir)
    for relative_path in relative_paths:
        if relative_path.endswith(".json"):
            db_json_path = os.path.join(db_dir, relative_path)
            try:
                with open(db_json_path, "r") as f:
                    data = json.load(f)
                # check if all necessary fields are present
                required_fields = ["sha512_hash", "clip_encoding"]
                if not all(field in data for field in required_fields):
                    raise ValueError("Missing required fields")

                database[relative_path] = data
            except Exception as e:
                print(f"Error loading database file {db_json_path}: {e}")
    return database


@click.command()
@click.option(
    "--image-dir",
    "-i",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing images to process.",
)
@click.option("--db-dir", "-d", required=True, type=click.Path(file_okay=False), help="Directory to store the database JSON files.")
def main(image_dir: str, db_dir: str):
    """CLI to update the image database."""
    os.makedirs(db_dir, exist_ok=True)

    def dummy_process_image(image_path: str, db_json_path: str):
        # Placeholder for actual image processing logic
        with open(db_json_path, "w") as f:
            f.write(f"Processed {image_path}\n")

    update_database(image_dir, db_dir, dummy_process_image)


if __name__ == "__main__":
    main()
