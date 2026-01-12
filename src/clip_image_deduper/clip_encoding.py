#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""open_clip model and preprocessing setup for clip_image_deduper."""
import open_clip
import torch
from typing import List
import PIL.Image
import numpy as np
import click
import tqdm

# default model
default_model_id = "hf-hub:timm/PE-Core-bigG-14-448"


class CLIPImageEncoder:
    """Class to handle CLIP model and preprocessing."""

    def __init__(self, model_id: str = default_model_id, device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_id, device=device, jit=True, precision="fp32")
        self.tokenizer = open_clip.get_tokenizer(model_id)

    def encode_image(self, image: PIL.Image.Image) -> np.ndarray:
        """Encode an image using the CLIP model."""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)  # Add batch dimension and move to device
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        return image_features.cpu().squeeze(0).numpy()  # Remove batch dimension and move to CPU


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two feature tensors."""
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
    similarity = np.dot(a_norm, b_norm.T) / (np.linalg.norm(a_norm) * np.linalg.norm(b_norm))
    # normalize to [-1, 1]
    return similarity


@click.command()
@click.option("--model-id", "-m", default=default_model_id, help="CLIP model identifier.", show_default=True)
@click.option("--device", "-d", default="cpu", help="Device to run the model on.", show_default=True)
@click.argument("image_paths", nargs=-1, type=click.Path(exists=True, dir_okay=False))
def main(model_id: str, device: str, image_paths: List[str]):
    """CLI to encode images using CLIP model."""
    encoder = CLIPImageEncoder(model_id=model_id, device=device)
    print(f"Loaded model: {model_id} on device: {device}")
    t = tqdm.tqdm(image_paths)
    for image_path in t:
        try:
            image = PIL.Image.open(image_path).convert("RGB")
            features = encoder.encode_image(image)
            t.write(f"Encoded features for the image {image_path}: {features}")
        except Exception as e:
            t.write(f"Error processing image {image_path}: {e}")


if __name__ == "__main__":
    main()
