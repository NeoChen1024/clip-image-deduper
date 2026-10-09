#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This module is for updating the "database" of clip_image_deduper. It scans a specified directory for valid images
# and (re)encodes every image whose mtime differs from the one recorded in the SQLite embedding database.

import multiprocessing as mp
import os
import queue
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Thread
from typing import Callable, List, Tuple, Union

import click
import numpy as np
import PIL.Image
import torch
import tqdm

from .clip_encoding import CLIPImageEncoder, default_model_id
from .db_store import EmbeddingDB, ModelMismatchError


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


def _load_and_prepare_image(
    preprocessor: Callable, image_dir: str, relative_path: str
) -> Tuple[str, Union[np.ndarray, Exception]]:
    """Load and validate a single image, returning the preprocessed tensor as numpy or an Exception."""
    image_path = os.path.join(image_dir, relative_path)

    try:
        if not verify_image(image_path):
            raise ValueError("Invalid image")

        with PIL.Image.open(image_path) as img:
            img = img.convert("RGB")
            img.load()

        return relative_path, preprocessor(img).numpy()
    except Exception as e:
        return relative_path, e


def _encode_and_save_batch(
    encoder: CLIPImageEncoder,
    db: EmbeddingDB,
    image_dir: str,
    batch_paths: List[str],
    image_np_batch: List[np.ndarray],
) -> None:
    if not image_np_batch:
        return

    embeddings = encoder.encode_images([torch.from_numpy(arr) for arr in image_np_batch])
    rows = []
    for rel_path, embedding in zip(batch_paths, embeddings):
        st = os.stat(os.path.join(image_dir, rel_path))
        rows.append((rel_path, st.st_mtime, st.st_size, embedding))
    db.upsert(rows)


def _init_worker():
    torch.set_num_threads(1)


def update_database(
    encoder: CLIPImageEncoder,
    image_dir: str,
    db_path: str,
    force_update: bool = False,
    clean_orphans: bool = True,
    batch_size: int = 4,
):
    """Update the embedding database by encoding every new or modified image under ``image_dir``."""
    with EmbeddingDB(db_path) as db:
        try:
            db.ensure_model(encoder.model_id, reset_on_mismatch=force_update)
        except ModelMismatchError as e:
            raise click.ClickException(str(e)) from e

        index = db.load_index()  # relative_path -> mtime at encoding time
        seen: set = set()
        candidates: List[str] = []
        for relative_path in walk_directory_relative(image_dir):
            seen.add(relative_path)
            try:
                image_mtime = os.path.getmtime(os.path.join(image_dir, relative_path))
            except OSError:
                # Ignore pathological filesystem issues here; they will be surfaced later if needed.
                continue
            if not force_update and index.get(relative_path) == image_mtime:
                continue
            candidates.append(relative_path)

        if candidates:
            _encode_candidates(encoder, db, image_dir, candidates, batch_size)

        if clean_orphans:
            orphans = [p for p in index if p not in seen]
            if orphans:
                for p in orphans:
                    tqdm.tqdm.write(f"Removing orphaned database entry: {p}")
                db.delete(orphans)


def _encode_candidates(encoder: CLIPImageEncoder, db: EmbeddingDB, image_dir: str, candidates: List[str], batch_size: int):
    """Asynchronously load and validate candidate images, then encode and store them in batches."""
    t = tqdm.tqdm(total=len(candidates))
    result_queue: queue.Queue = queue.Queue(maxsize=max(batch_size * 2, 1))
    preprocessor = encoder.get_preprocessor()

    max_workers = min(batch_size, os.cpu_count() or 1)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx, initializer=_init_worker) as executor:

        def _producer(preprocessor, image_dir, candidates, q):
            max_futures = q.maxsize
            for start in range(0, len(candidates), max_futures):
                chunk = candidates[start : start + max_futures]
                futures = [executor.submit(_load_and_prepare_image, preprocessor, image_dir, p) for p in chunk]
                for future in as_completed(futures):
                    q.put(future.result())

        Thread(target=_producer, args=(preprocessor, image_dir, candidates, result_queue), daemon=True).start()

        batch_paths: List[str] = []
        image_np_batch: List[np.ndarray] = []
        done = 0
        while done < len(candidates):
            rel_path, result = result_queue.get()

            if isinstance(result, Exception):
                image_path = os.path.join(image_dir, rel_path)
                t.write(f"Skipping: {image_path} ({result})")
            else:
                batch_paths.append(rel_path)
                image_np_batch.append(result)

                if len(image_np_batch) >= batch_size:
                    _encode_and_save_batch(encoder, db, image_dir, batch_paths, image_np_batch)
                    batch_paths.clear()
                    image_np_batch.clear()

            done += 1
            t.update()

        if image_np_batch:
            _encode_and_save_batch(encoder, db, image_dir, batch_paths, image_np_batch)


def load_database(db_path: str) -> Tuple[List[str], np.ndarray]:
    """Load every embedding from the database as ``(relative_paths, (N, D) float32 array)``."""
    if not os.path.exists(db_path):
        return [], np.empty((0, 0), dtype=np.float32)
    with EmbeddingDB(db_path) as db:
        return db.load_all()


@click.command()
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing images to process.",
)
@click.option(
    "--db",
    "-d",
    type=click.Path(dir_okay=False),
    required=True,
    help="SQLite file storing the embedding database.",
)
@click.option(
    "--clean-orphans/--no-clean-orphans",
    default=True,
    help="Whether to remove database entries whose images no longer exist.",
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
    db: str,
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
        update_database(encoder, image_dir, db, force_update, clean_orphans, batch_size=batch_size)
    print("Try loading the database...")
    paths, embeddings = load_database(db)
    print(f"Loaded {len(paths)} entries in the database, shape {embeddings.shape}.")


if __name__ == "__main__":
    main()
