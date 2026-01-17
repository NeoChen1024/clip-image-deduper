#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""open_clip model and preprocessing setup for clip_image_deduper."""
from typing import List, Optional, overload

import click
import numpy as np
import open_clip
import PIL.Image
import torch

from .similarity import euclidean_distance

# default model
default_model_id = "hf-hub:timm/PE-Core-bigG-14-448"


class CLIPImageEncoder:
    """Class to handle CLIP model and preprocessing."""

    def __init__(self, model_id: str = default_model_id, device: str = "cpu", dtype: Optional[str] = None):
        self.model_id = model_id
        self.device = device
        if dtype is None:
            if "cuda" in device:
                dtype = "fp16"
            else:
                dtype = "fp32"
        # Open CLIP uses "fp32", "fp16" as dtype strings
        elif dtype == "float32":
            dtype = "fp32"
        elif dtype == "float16":
            dtype = "fp16"
        tdtype = torch.float32
        if dtype == "fp16":
            tdtype = torch.float16
        self.tdtype = tdtype
        print(f"Using CLIP model: {model_id} on device: {device}, dtype: {dtype}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_id, device=device, jit=True, precision=dtype)
        self.tokenizer = open_clip.get_tokenizer(model_id)

    # unload model from memory when done
    def __del__(self):
        self.model.to("cpu")  # move to cpu before deleting
        del self.model
        torch.cuda.empty_cache()

    def get_preprocessor(self):
        """Get the preprocessing function. (PIL.Image -> torch.Tensor), to maximize performance."""
        return self.preprocess


    @torch.no_grad()
    @torch.compile()
    def preprocess_encode_images(self, images: List[PIL.Image.Image]) -> np.ndarray:
        """Encode a list of images using the CLIP model."""
        # TODO: performance can be improved by doing preprocessing asynchronously in parallel
        image_inputs = torch.stack([self.preprocess(img) for img in images]).to(self.tdtype).to(self.device)  # bottleneck here
        image_features = self.model.encode_image(image_inputs)
        return image_features.cpu().float().numpy()

    @torch.no_grad()
    @torch.compile()
    def encode_images(self, preprocessed_image_tensor: List[torch.Tensor]) -> np.ndarray:
        """Encode a list of images using the CLIP model."""
        image_features = self.model.encode_image(torch.stack(preprocessed_image_tensor).to(self.tdtype).to(self.device))
        return image_features.cpu().float().numpy()


@click.command()
@click.option("--model-id", "-m", default=default_model_id, help="CLIP model identifier.", show_default=True)
@click.option("--device", "-d", default="cpu", help="Device to run the model on.", show_default=True)
@click.option(
    "--dtype",
    "-p",
    type=click.Choice(["float32", "float16", "bfloat16"], case_sensitive=False),
    default="float32",
    help="Data type for model precision. Default is float16 for CUDA and float32 otherwise.",
    show_default=True,
)
@click.argument("image_paths", nargs=-1, type=click.Path(exists=True, dir_okay=False))
def main(model_id: str, device: str, dtype: str, image_paths: List[str]):
    """CLI to encode images using CLIP model."""
    encoder = CLIPImageEncoder(model_id=model_id, device=device, dtype=dtype)
    print(f"Loaded model: {model_id} on device: {device}")
    features = np.ndarray([])
    images_pil = []
    for image_path in image_paths:
        try:
            image = PIL.Image.open(image_path).convert("RGB")
            images_pil.append(image)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    if images_pil:
        features = encoder.preprocess_encode_images(images_pil)
        for img_path, feat in zip(image_paths, features):
            print(f"Image: {img_path}, Feature shape: {feat.shape}")

    # show similarity matrix
    print("Feature matrix shape:", features.shape)
    distance_matrix = euclidean_distance(features, features)
    print("Euclidean Distance Matrix:")
    print(distance_matrix)


if __name__ == "__main__":
    main()
