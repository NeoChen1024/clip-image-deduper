"""open_clip model wrapper used to turn preprocessed images into embeddings."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np
import open_clip
import PIL.Image
import torch

logger = logging.getLogger(__name__)

# SigLIP2 so400m: embeddings of a re-encoded / rescaled copy stay within ~0.5-3 of the original while different
# images sit at 5+ (norm ~16). PE-Core-bigG was tried first but is so sensitive to compression artifacts that a JPEG
# q90 re-save of the same picture lands 5-15 away, too close to the distance between different pictures.
default_model_id = "hf-hub:timm/ViT-SO400M-16-SigLIP2-512"

# Accepted spellings for the precision option -> (open_clip precision string, torch dtype)
_PRECISIONS: dict[str, tuple[str, torch.dtype]] = {
    "fp32": ("fp32", torch.float32),
    "float32": ("fp32", torch.float32),
    "fp16": ("fp16", torch.float16),
    "float16": ("fp16", torch.float16),
}
precision_choices = tuple(_PRECISIONS)


class CLIPImageEncoder:
    """Loads an open_clip model once and encodes batches of preprocessed images.

    ``dtype`` defaults to fp16 on CUDA and fp32 elsewhere. Note that fp16 inference is what makes the stored fp16
    embeddings lossless: every value the model emits is already an fp16 number.
    """

    def __init__(self, model_id: str = default_model_id, device: str = "cpu", dtype: str | None = None):
        if dtype is None:
            dtype = "fp16" if "cuda" in device else "fp32"
        try:
            precision, self.tdtype = _PRECISIONS[dtype.lower()]
        except KeyError:
            raise ValueError(f"Unknown dtype {dtype!r}; expected one of {precision_choices}") from None
        self.model_id = model_id
        self.device = device
        logger.info("Loading CLIP model %s on %s (%s)", model_id, device, precision)
        # jit=False: TorchScript brings nothing for a plain encode_image() call and fails to script some models' text towers.
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_id, device=device, jit=False, precision=precision)
        self.model.eval()

    def get_preprocessor(self) -> Callable[[PIL.Image.Image], torch.Tensor]:
        """The PIL -> tensor transform, picklable so worker processes can run it."""
        return self.preprocess

    @torch.no_grad()
    def encode_images(self, preprocessed: Sequence[torch.Tensor]) -> np.ndarray:
        """Encode already-preprocessed image tensors. Returns an ``(N, D)`` float32 array."""
        batch = torch.stack(list(preprocessed)).to(self.device, self.tdtype)
        return self.model.encode_image(batch).float().cpu().numpy()

    @torch.no_grad()
    def encode_pil_images(self, images: Sequence[PIL.Image.Image]) -> np.ndarray:
        """Preprocess and encode PIL images in one go (convenience for small inputs)."""
        return self.encode_images([self.preprocess(img.convert("RGB")) for img in images])

    def close(self) -> None:
        """Drop the model and free device memory."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
