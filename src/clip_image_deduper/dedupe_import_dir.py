#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Remove duplicate images from a "import directory" by moving them to a trash directory.

import gc
import os
from shutil import move
from typing import Any, List, Optional, Tuple

import click
import humanize
import numpy as np
import torch
import tqdm

from .clip_encoding import CLIPImageEncoder, default_model_id
from .db_processing import load_database, update_database
from .similarity import (
    DistanceIndex,
    default_euclidean_distance_threshold,
    find_close_pairs_cross,
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
    "--base-db",
    "-bd",
    type=click.Path(dir_okay=False),
    required=True,
    help="SQLite file storing the base embedding database.",
)
@click.option(
    "--import-image-dir",
    "-ii",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing import images to process.",
)
@click.option(
    "--import-db",
    "-id",
    type=click.Path(dir_okay=False),
    required=True,
    help="SQLite file storing the import embedding database.",
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
    "--batch-size",
    "-b",
    type=int,
    default=4,
    help="Batch size for processing images when updating the database.",
    show_default=True,
)
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
    base_db: str,
    import_image_dir: str,
    import_db: str,
    model_id: str,
    force_update: bool,
    clean_orphans: bool,
    device: str,
    skip_update: bool,
    dry_run: bool,
    threshold: float,
    trash_dir: str,
    batch_size: int = 4,
):
    torch.set_float32_matmul_precision("highest")
    if not skip_update:
        # Dry-run still needs a fresh DB to produce an accurate preview.
        # Callers that need a no-write run can combine --dry-run with
        # --skip-update and accept potentially stale embeddings.
        encoder = CLIPImageEncoder(model_id=model_id, device=device)
        print("Updating base database...")
        update_database(encoder, base_image_dir, base_db, force_update, clean_orphans, batch_size=batch_size)
        print("Updating import database...")
        update_database(encoder, import_image_dir, import_db, force_update, clean_orphans, batch_size=batch_size)
        encoder.cleanup()
        del encoder
        gc.collect()

    def db_processing(db_type: str, db_path: str) -> Tuple[List[str], DistanceIndex]:
        print(f"Loading {db_type} database...")
        image_paths, embeddings_db = load_database(db_path)  # (N, D)
        print(f"Loaded {len(image_paths)} entries in the database.")
        if len(image_paths) == 0:
            print(f"No entries found in the {db_type} database. Exiting.")
            raise SystemExit(1)

        index = DistanceIndex(embeddings_db, device)
        print(
            f"Embeddings DB shape of {db_type}: ({index.n}, {index.dim}), device memory: "
            f"{humanize.naturalsize(index.nbytes, binary=True)}, backend: {index.backend_name()}"
        )
        return image_paths, index

    base_image_paths, base_index = db_processing("base", base_db)
    import_image_paths, import_index = db_processing("import", import_db)

    print("Finding duplicates...")
    duplicate_count = 0
    t = tqdm.tqdm(total=len(import_image_paths), desc="Processing import images", unit="image")

    for start, end in import_index.iter_blocks(other=base_index):
        ii, jj, dd = find_close_pairs_cross(import_index, base_index, threshold, start, end)
        if len(ii):
            order = np.lexsort((jj, ii))
            ii, jj, dd = ii[order], jj[order], dd[order]
            for i in np.unique(ii):
                sel = ii == i
                image_path = import_image_paths[int(i)]
                matches = [(base_image_paths[int(j)], float(d)) for j, d in zip(jj[sel], dd[sel])]
                t.write(f"Found {len(matches)} instances for {image_path}: {matches}")
                duplicate_count += len(matches)
                if trash_dir is not None:
                    move_duplicate(image_path, import_image_dir, trash_dir, dry_run, t)
        t.update(end - start)
    t.close()

    dry_run_str = ""
    if dry_run:
        dry_run_str = " (dry run, no files were moved)"

    print(f"Deduplication complete., processed {len(import_image_paths)} images, found {duplicate_count} duplicates.{dry_run_str}")

    base_index.release()
    import_index.release()
    del base_index, import_index
    gc.collect()
    torch.cuda.empty_cache()
    torch.compiler.reset()
