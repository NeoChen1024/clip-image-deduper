"""Incremental encoding: find images that are new or changed, decode them in worker processes, encode on the model
device in batches, and store the embeddings plus file metadata in the database.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait

import numpy as np
import PIL.Image
import PIL.ImageFile
import torch

from .db_store import EmbeddingDB, ImageRecord, ModelMismatchError
from .encoder import CLIPImageEncoder, PendingBatch

logger = logging.getLogger(__name__)

# This is a tool for the user's own collection, not a server decoding untrusted uploads: behave like an image viewer.
# Pillow's defaults reject images above ~179 Mpixel ("decompression bomb") and files whose data ends early, both of
# which ordinary viewers display without complaint. These module globals apply to the process that imports this
# module; worker processes re-import it, so they inherit the same settings.
PIL.Image.MAX_IMAGE_PIXELS = None
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

# Extensions Pillow knows how to open. Anything else in the image dir (sidecar .txt, .json, ...) is ignored silently.
IMAGE_EXTENSIONS = frozenset(ext for ext, fmt in PIL.Image.registered_extensions().items() if fmt in PIL.Image.OPEN)


def walk_directory_relative(directory: str) -> Iterator[str]:
    """Yield file paths relative to ``directory``, depth first."""
    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.relpath(os.path.join(root, file), directory)


def is_image_path(relative_path: str) -> bool:
    return os.path.splitext(relative_path)[1].lower() in IMAGE_EXTENSIONS


def _load_image(
    preprocessor: Callable, array_dtype: np.dtype, image_dir: str, relative_path: str
) -> tuple[str, tuple[ImageRecord, np.ndarray] | Exception]:
    """Worker: decode one image and run the model's full preprocessing (resize, normalize, dtype cast).

    The main process gets back a ready-to-stack ``(C, H, W)`` array in the model's dtype, so all it has to do for
    the GPU is gather a batch and copy it over. Returns the record + array, or the exception.

    Any failure here means "skip this file", so the catch is deliberately broad: a corrupt file must not take the
    whole run down, and Pillow raises a wide variety of exception types for broken inputs.
    """
    image_path = os.path.join(image_dir, relative_path)
    try:
        st = os.stat(image_path)
        with PIL.Image.open(image_path) as img:
            fmt = img.format or os.path.splitext(relative_path)[1].lstrip(".").upper()
            width, height = img.size
            array = preprocessor(img.convert("RGB")).numpy().astype(array_dtype, copy=False)  # animated: first frame
        record = ImageRecord(relative_path, st.st_mtime, st.st_size, width, height, fmt)
        return relative_path, (record, array)
    except Exception as e:  # noqa: BLE001 - see docstring
        return relative_path, e


def _init_worker() -> None:
    torch.set_num_threads(1)


def find_candidates(image_dir: str, index: dict[str, float], *, force: bool = False) -> tuple[list[str], set[str]]:
    """Images under ``image_dir`` that need (re)encoding, and the set of all image paths seen.

    An image needs encoding when it has no row or its mtime differs from the stored one (``!=`` rather than ``>``,
    so a file replaced by an older copy is re-encoded too).
    """
    candidates: list[str] = []
    seen: set[str] = set()
    for rel in walk_directory_relative(image_dir):
        if not is_image_path(rel):
            continue
        seen.add(rel)
        try:
            mtime = os.path.getmtime(os.path.join(image_dir, rel))
        except OSError as e:
            logger.warning("Cannot stat %s: %s", rel, e)
            continue
        if force or index.get(rel) != mtime:
            candidates.append(rel)
    return candidates, seen


def default_workers() -> int:
    return os.cpu_count() or 1


def encode_images(
    encoder: CLIPImageEncoder,
    db: EmbeddingDB,
    image_dir: str,
    candidates: Sequence[str],
    *,
    batch_size: int,
    workers: int | None = None,
    progress: Callable[[int], object] | None = None,
) -> int:
    """Decode ``candidates`` in ``workers`` processes, encode in batches of ``batch_size`` and store. Returns #stored.

    Three stages overlap:

    * ``workers`` processes decode and fully preprocess images. The number of workers is independent of the batch
      size; it is a function of how fast the CPU can decode, the batch size of how much VRAM there is.
    * The main thread keeps ``2 * workers`` decode jobs in flight (so no worker ever waits for the main thread),
      gathers finished arrays into batches and submits each batch to the device without waiting for it.
    * The device runs the previous batch while the main thread is back to collecting decoded images; the result is
      fetched and written to the database only when the next batch has been submitted (or at the end).

    Memory: at most ``2 * workers`` preprocessed arrays plus two batches live at any time.
    """
    if not candidates:
        return 0
    if workers is None:
        workers = default_workers()
    workers = max(1, workers)
    window = 2 * workers
    preprocessor = encoder.get_preprocessor()
    array_dtype = encoder.array_dtype
    stored = 0
    batch: list[tuple[ImageRecord, np.ndarray]] = []
    in_flight: tuple[list[ImageRecord], PendingBatch] | None = None

    def finish() -> None:
        nonlocal stored, in_flight
        if in_flight is None:
            return
        records, pending = in_flight
        in_flight = None
        embeddings = encoder.collect(pending)
        db.upsert(zip(records, embeddings))
        stored += len(records)

    def flush() -> None:
        nonlocal in_flight
        if not batch:
            return
        records = [rec for rec, _ in batch]
        pending = encoder.submit([arr for _, arr in batch])
        batch.clear()
        finish()  # the previous batch has had the whole decode interval to complete
        in_flight = (records, pending)

    todo = iter(candidates)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init_worker) as pool:
        pending_jobs: set[Future] = set()

        def refill() -> None:
            for rel in todo:
                pending_jobs.add(pool.submit(_load_image, preprocessor, array_dtype, image_dir, rel))
                if len(pending_jobs) >= window:
                    break

        refill()
        while pending_jobs:
            done, pending_jobs = wait(pending_jobs, return_when=FIRST_COMPLETED)
            for future in done:
                rel, result = future.result()
                if isinstance(result, Exception):
                    logger.warning("Skipping %s: %s", os.path.join(image_dir, rel), result)
                else:
                    batch.append(result)
                    if len(batch) >= batch_size:
                        flush()
                if progress is not None:
                    progress(1)
            refill()
    flush()
    finish()
    return stored


def update_database(
    encoder: CLIPImageEncoder,
    image_dir: str,
    db_path: str,
    *,
    force_update: bool = False,
    clean_orphans: bool = True,
    batch_size: int = 4,
    workers: int | None = None,
    progress_factory: Callable[[int], Callable[[int], object]] | None = None,
) -> None:
    """Bring ``db_path`` up to date with the images under ``image_dir``.

    ``workers`` is the number of decoder processes (default: one per CPU). ``progress_factory(total)`` may return a
    callable that is invoked with increments as images are processed.
    """
    with EmbeddingDB(db_path) as db:
        try:
            db.ensure_model(encoder.model_id, reset_on_mismatch=force_update)
        except ModelMismatchError as e:
            raise RuntimeError(str(e)) from e

        index = db.load_index()
        candidates, seen = find_candidates(image_dir, index, force=force_update)
        logger.info("%d images, %d to encode", len(seen), len(candidates))
        if candidates:
            progress = progress_factory(len(candidates)) if progress_factory else None
            encode_images(encoder, db, image_dir, candidates, batch_size=batch_size, workers=workers, progress=progress)

        if clean_orphans:
            orphans = [p for p in index if p not in seen]
            for p in orphans:
                logger.info("Removing orphaned database entry: %s", p)
            db.delete(orphans)
