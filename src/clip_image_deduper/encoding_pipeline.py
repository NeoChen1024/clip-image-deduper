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
from .encoder import CLIPImageEncoder

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


def _load_image(preprocessor: Callable, image_dir: str, relative_path: str) -> tuple[str, tuple[ImageRecord, np.ndarray] | Exception]:
    """Worker: decode one image and run the model's preprocessing. Returns the record + tensor, or the exception.

    Any failure here means "skip this file", so the catch is deliberately broad: a corrupt file must not take the
    whole run down, and Pillow raises a wide variety of exception types for broken inputs.
    """
    image_path = os.path.join(image_dir, relative_path)
    try:
        st = os.stat(image_path)
        with PIL.Image.open(image_path) as img:
            fmt = img.format or os.path.splitext(relative_path)[1].lstrip(".").upper()
            width, height = img.size
            tensor = preprocessor(img.convert("RGB")).numpy()  # animated formats: first frame
        record = ImageRecord(relative_path, st.st_mtime, st.st_size, width, height, fmt)
        return relative_path, (record, tensor)
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


def encode_images(
    encoder: CLIPImageEncoder,
    db: EmbeddingDB,
    image_dir: str,
    candidates: Sequence[str],
    *,
    batch_size: int,
    progress: Callable[[int], object] | None = None,
) -> int:
    """Decode ``candidates`` in worker processes, encode in batches of ``batch_size`` and store. Returns #stored.

    Decoding runs ahead of the GPU with a bounded window of in-flight futures, so memory stays bounded and the model
    never waits on a single slow file. No helper thread: the main thread alternates between waiting for decoded
    images and running the model.
    """
    if not candidates:
        return 0
    preprocessor = encoder.get_preprocessor()
    max_workers = max(1, min(batch_size, os.cpu_count() or 1))
    window = max(batch_size * 2, max_workers)
    stored = 0
    batch: list[tuple[ImageRecord, np.ndarray]] = []

    def flush() -> None:
        nonlocal stored
        if not batch:
            return
        embeddings = encoder.encode_images([torch.from_numpy(t) for _, t in batch])
        db.upsert((rec, emb) for (rec, _), emb in zip(batch, embeddings))
        stored += len(batch)
        batch.clear()

    todo = iter(candidates)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn"), initializer=_init_worker) as pool:
        pending: set[Future] = set()

        def refill() -> None:
            for rel in todo:
                pending.add(pool.submit(_load_image, preprocessor, image_dir, rel))
                if len(pending) >= window:
                    break

        refill()
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
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
    return stored


def update_database(
    encoder: CLIPImageEncoder,
    image_dir: str,
    db_path: str,
    *,
    force_update: bool = False,
    clean_orphans: bool = True,
    batch_size: int = 4,
    progress_factory: Callable[[int], Callable[[int], object]] | None = None,
) -> None:
    """Bring ``db_path`` up to date with the images under ``image_dir``.

    ``progress_factory(total)`` may return a callable that is invoked with increments as images are processed.
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
            encode_images(encoder, db, image_dir, candidates, batch_size=batch_size, progress=progress)

        if clean_orphans:
            orphans = [p for p in index if p not in seen]
            for p in orphans:
                logger.info("Removing orphaned database entry: %s", p)
            db.delete(orphans)
