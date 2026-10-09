"""timm image tower wrapper used to turn preprocessed images into embeddings.

Only the image tower of a CLIP/SigLIP model is needed for deduplication, so the model is loaded straight from timm
(`timm.create_model`), which is also what open_clip uses under the hood for these architectures. Embeddings are
bit-identical to open_clip's ``encode_image`` for the same checkpoint, at 40% of the download size.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import PIL.Image
import timm
import timm.data
import torch

logger = logging.getLogger(__name__)

# SigLIP2 so400m, 512px: embeddings of a re-encoded / rescaled copy stay within ~0.5-3 of the original while different
# images sit at 5+ (norm ~16). PE-Core-bigG (timm: vit_pe_core_bigG_14_448.fb) was tried first but is so sensitive to
# compression artifacts that a JPEG q90 re-save of the same picture lands 5-15 away.
default_model_id = "vit_so400m_patch16_siglip_512.v2_webli"

# Accepted spellings for the precision option -> torch dtype
_PRECISIONS: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}
precision_choices = tuple(_PRECISIONS)


@dataclass(slots=True)
class PendingBatch:
    """A forward pass that has been queued on the device but not synchronized yet (see ``CLIPImageEncoder.submit``).

    ``host`` is the pinned staging buffer; it is kept alive here because the asynchronous copy may still be reading
    from it.
    """

    out: torch.Tensor
    n: int
    host: torch.Tensor


class CLIPImageEncoder:
    """Loads a timm image tower once and encodes batches of preprocessed images.

    ``dtype`` defaults to fp16 on CUDA and fp32 elsewhere. fp16 inference is what makes the stored fp16 embeddings
    lossless: every value the model emits is already an fp16 number.

    ``compile=True`` wraps the forward in ``torch.compile`` (about 10% faster on an RTX 4080; the model is compute
    bound). Compiled graphs are shape-specialized, so every batch is padded to ``batch_size`` to avoid recompiles.
    """

    def __init__(
        self,
        model_id: str = default_model_id,
        device: str = "cpu",
        dtype: str | None = None,
        *,
        compile: bool = False,
        batch_size: int | None = None,
    ):
        if dtype is None:
            dtype = "fp16" if "cuda" in device else "fp32"
        try:
            self.tdtype = _PRECISIONS[dtype.lower()]
        except KeyError:
            raise ValueError(f"Unknown dtype {dtype!r}; expected one of {precision_choices}") from None
        self.model_id = model_id
        self.device = device
        logger.info("Loading image model %s on %s (%s%s)", model_id, device, dtype, ", compiled" if compile else "")
        self.model = timm.create_model(model_id, pretrained=True, num_classes=0).to(device=device, dtype=self.tdtype).eval()

        # CLIP-style models are trained on the whole image squashed to the input size. timm's default eval transform
        # center-crops 90%, which would shift every embedding; override to the exact open_clip preprocessing.
        cfg = timm.data.resolve_data_config({}, model=self.model)
        cfg.update(crop_pct=1.0, crop_mode="squash")
        self.preprocess = timm.data.create_transform(**cfg, is_training=False)
        self.input_size = tuple(cfg["input_size"])

        self._compiled_batch = None
        self._forward: Callable[[torch.Tensor], torch.Tensor] = self.model
        if compile:
            if batch_size is None or batch_size < 1:
                raise ValueError("compile=True needs a fixed batch_size")
            self._compiled_batch = batch_size
            self._forward = torch.compile(self.model, dynamic=False)

    def get_preprocessor(self) -> Callable[[PIL.Image.Image], torch.Tensor]:
        """The PIL -> tensor transform, picklable so worker processes can run it."""
        return self.preprocess

    @property
    def array_dtype(self) -> np.dtype:
        """numpy dtype loaders should hand over: the model's own dtype so the host->device copy is as small as
        possible, except bf16 which numpy lacks (fp32 then, cast on the device)."""
        return np.dtype(np.float16) if self.tdtype == torch.float16 else np.dtype(np.float32)

    @torch.no_grad()
    def submit(self, preprocessed: Sequence[np.ndarray]) -> PendingBatch:
        """Queue one forward pass on the device and return without waiting for it.

        The arrays must already be fully preprocessed (``get_preprocessor`` output, ``(C, H, W)``), ideally in
        ``array_dtype``. They are gathered into one pinned host buffer, copied asynchronously and run; call
        ``collect`` for the result. On CUDA this lets the caller go back to feeding the decoders while the GPU works.
        """
        n = len(preprocessed)
        if n == 0:
            raise ValueError("Empty batch")
        rows = n
        if self._compiled_batch is not None:
            if n > self._compiled_batch:
                raise ValueError(f"Got {n} images but the compiled model is fixed at batch_size={self._compiled_batch}")
            rows = self._compiled_batch
        pin = self.device.startswith("cuda")
        host = torch.empty((rows, *preprocessed[0].shape), dtype=self.tdtype, pin_memory=pin)
        for i, a in enumerate(preprocessed):
            host[i].copy_(torch.from_numpy(np.ascontiguousarray(a)))
        if rows > n:
            host[n:].zero_()
        batch = host.to(self.device, non_blocking=pin)
        out = self._forward(batch)[:n]
        return PendingBatch(out, n, host)

    def collect(self, pending: PendingBatch) -> np.ndarray:
        """Wait for a submitted batch and return its ``(N, D)`` float32 embeddings."""
        return pending.out.float().cpu().numpy()

    def encode_images(self, preprocessed: Sequence[torch.Tensor | np.ndarray]) -> np.ndarray:
        """Encode already-preprocessed image tensors synchronously. Returns an ``(N, D)`` float32 array."""
        arrays = [t.numpy() if isinstance(t, torch.Tensor) else t for t in preprocessed]
        return self.collect(self.submit(arrays))

    def encode_pil_images(self, images: Sequence[PIL.Image.Image]) -> np.ndarray:
        """Preprocess and encode PIL images (convenience for small inputs; respects the compiled batch size)."""
        arrays = [self.preprocess(img.convert("RGB")).to(self.tdtype if self.tdtype != torch.bfloat16 else torch.float32).numpy() for img in images]
        step = self._compiled_batch or len(arrays) or 1
        chunks = [self.encode_images(arrays[i : i + step]) for i in range(0, len(arrays), step)]
        return np.concatenate(chunks) if chunks else np.empty((0, 0), dtype=np.float32)

    def close(self) -> None:
        """Drop the model and free device memory."""
        del self._forward
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
