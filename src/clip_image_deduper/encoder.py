"""timm image tower wrapper used to turn preprocessed images into embeddings.

Only the image tower of a CLIP/SigLIP model is needed for deduplication, so the model is loaded straight from timm
(`timm.create_model`), which is also what open_clip uses under the hood for these architectures. Embeddings are
bit-identical to open_clip's ``encode_image`` for the same checkpoint, at 40% of the download size.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

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

    @torch.no_grad()
    def encode_images(self, preprocessed: Sequence[torch.Tensor]) -> np.ndarray:
        """Encode already-preprocessed image tensors. Returns an ``(N, D)`` float32 array."""
        n = len(preprocessed)
        batch = torch.stack(list(preprocessed)).to(self.device, self.tdtype)
        if self._compiled_batch is not None and n != self._compiled_batch:
            if n > self._compiled_batch:
                raise ValueError(f"Got {n} images but the compiled model is fixed at batch_size={self._compiled_batch}")
            pad = torch.zeros((self._compiled_batch - n, *batch.shape[1:]), device=batch.device, dtype=batch.dtype)
            batch = torch.cat([batch, pad])
        return self._forward(batch)[:n].float().cpu().numpy()

    @torch.no_grad()
    def encode_pil_images(self, images: Sequence[PIL.Image.Image]) -> np.ndarray:
        """Preprocess and encode PIL images (convenience for small inputs; respects the compiled batch size)."""
        tensors = [self.preprocess(img.convert("RGB")) for img in images]
        step = self._compiled_batch or len(tensors) or 1
        chunks = [self.encode_images(tensors[i : i + step]) for i in range(0, len(tensors), step)]
        return np.concatenate(chunks) if chunks else np.empty((0, 0), dtype=np.float32)

    def close(self) -> None:
        """Drop the model and free device memory."""
        del self._forward
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
