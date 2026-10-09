"""Background jobs and image loading for the review GUI.

Disk reads never happen on the GUI thread: a :class:`Job` wraps a callable in a ``QRunnable`` and reports back
through signals, as in the dataset editor this GUI is modelled on.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable

import numpy as np
import PIL.Image
import PIL.ImageFile
from PySide6.QtCore import QObject, QRunnable, Qt, Signal
from PySide6.QtGui import QImage, QPixmap

# Same leniency as the encoder: this is the user's own collection, behave like an image viewer.
PIL.Image.MAX_IMAGE_PIXELS = None
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


class Signals(QObject):
    result = Signal(object)
    error = Signal(str)
    progress = Signal(int)


class Job(QRunnable):
    """Run ``function(signals)`` in the pool; its return value is emitted on ``signals.result``.

    ``connect`` wires both outcomes to the GUI thread with queued connections and keeps the job alive until the
    result has been delivered: a lambda slot has no receiver object, so without this Qt would run it on the worker
    thread, and an auto-deleted runnable could take its ``Signals`` down before the queued event arrives.
    """

    _alive: set[Job] = set()

    def __init__(self, function: Callable[[Signals], object]):
        super().__init__()
        self.function = function
        self.signals = Signals()

    def connect(self, result: Callable[[object], object], error: Callable[[str], object]) -> Job:
        self.signals.result.connect(lambda r: (Job._alive.discard(self), result(r)), Qt.ConnectionType.QueuedConnection)
        self.signals.error.connect(lambda e: (Job._alive.discard(self), error(e)), Qt.ConnectionType.QueuedConnection)
        Job._alive.add(self)
        return self

    def run(self) -> None:
        try:
            self.signals.result.emit(self.function(self.signals))
        except Exception as error:  # noqa: BLE001 - reported to the GUI
            self.signals.error.emit(str(error))


def load_image(path: str) -> PIL.Image.Image:
    with PIL.Image.open(path) as img:
        return img.convert("RGB")  # animated: first frame


def qimage(im: PIL.Image.Image) -> QImage:
    rgb = im.convert("RGB")
    return QImage(rgb.tobytes(), rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888).copy()


def thumbnail(path: str, size: tuple[int, int]) -> QImage:
    with PIL.Image.open(path) as img:
        img.draft("RGB", size)  # JPEG: decode at a reduced scale
        im = img.convert("RGB")
    im.thumbnail(size)
    return qimage(im)


def diff_image(a: PIL.Image.Image, b: PIL.Image.Image, gain: int = 4) -> QImage:
    """Absolute difference of ``a`` and ``b`` (``b`` resampled to ``a``'s size), amplified ``gain`` times."""
    if b.size != a.size:
        b = b.resize(a.size, PIL.Image.Resampling.LANCZOS)
    arr = np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16)) * gain
    out = PIL.Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    return qimage(out)


class LRU:
    """A small dict with least-recently-used eviction."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._d: OrderedDict = OrderedDict()

    def get(self, key):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key, value) -> None:
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.capacity:
            self._d.popitem(last=False)

    def __contains__(self, key) -> bool:
        return key in self._d

    def clear(self) -> None:
        self._d.clear()


__all__ = ["Job", "Signals", "LRU", "load_image", "qimage", "thumbnail", "diff_image", "QPixmap"]
