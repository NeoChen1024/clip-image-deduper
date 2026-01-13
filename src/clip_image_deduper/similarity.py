#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

default_euclidean_distance_threshold = 0.5


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the Euclidean distance matrix between two sets of vectors.

    Accepts a, b as (D,), (1, D), or (N, D)/(M, D).
    Returns shape (N, M).
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=-1)

@torch.no_grad()
def euclidean_distance_torch_1_to_many(
    a: np.ndarray,
    b_torch: torch.Tensor,
) -> np.ndarray:
    """
    a: (D,) or (1, D) numpy
    b_torch: (N, D) torch tensor already on correct device
    returns: (N,) numpy distances
    """
    a_t = torch.from_numpy(a).to(b_torch.device).float()
    if a_t.ndim == 1:
        a_t = a_t.unsqueeze(0)  # (1, D)

    # torch.cdist computes pairwise distances; result shape (1, N)
    d = torch.cdist(a_t, b_torch, p=2)
    return d[0].cpu().numpy()


def find_similar_images_euclidean(
    image_idx: int, image_embedding: np.ndarray, database: torch.Tensor, threshold: float = default_euclidean_distance_threshold
) -> List[Tuple[int, float]]:
    """Find similar images in the database based on Euclidean distance."""
    distances = euclidean_distance_torch_1_to_many(image_embedding, database)

    # For a single query vector vs database, distances has shape (1, N).
    # Squeeze to 1D so indexing and thresholding behave as expected.
    if distances.ndim == 2 and distances.shape[0] == 1:
        distances = distances[0]

    matches = np.where(distances <= threshold)[0]
    similar_images = [(int(idx), float(distances[idx])) for idx in matches if idx != image_idx]
    return similar_images
