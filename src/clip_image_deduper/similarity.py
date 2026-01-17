#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

default_euclidean_distance_threshold = 1  # same images will have distance 0.0 to 1 depending on encoding model, adjust as needed


# slow generic numpy version, for testing and reference
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
@torch.compile()
def euclidean_distance_torch_1_to_many(
    a: torch.Tensor,
    b: torch.Tensor,
) -> np.ndarray:
    """
    a: (1, D) torch tensor
    b: (N, D) torch tensor already on correct device
    returns: (N,) numpy distances
    """
    # torch.cdist computes pairwise distances; result shape (1, N)
    d = torch.cdist(a, b, p=2)
    return d[0].cpu().numpy()


@torch.compile()
def find_similar_images_euclidean(
    image_idx: int, image_embedding_1d: np.ndarray, database: torch.Tensor, threshold: float = default_euclidean_distance_threshold
) -> List[Tuple[int, float]]:
    """Find similar images in the database based on Euclidean distance.

    image_idx: index of the query image in the database, -1 if not in database
    image_embedding_1d: (D,) numpy array of the query image embedding
    database: (N, D) torch tensor of the database embeddings
    threshold: distance threshold for considering images as similar
    """
    image_embedding_unsqueezed = image_embedding_1d[np.newaxis, :]  # shape (1, D)
    image_embedding = torch.from_numpy(image_embedding_unsqueezed).to(database.device).float()
    distances = euclidean_distance_torch_1_to_many(image_embedding, database)

    # For a single query vector vs database, distances has shape (1, N).
    # Squeeze to 1D so indexing and thresholding behave as expected.
    if distances.ndim == 2 and distances.shape[0] == 1:
        distances = distances[0]

    matches = np.where(distances <= threshold)[0]
    if image_idx >= 0:
        similar_images = [(int(idx), float(distances[idx])) for idx in matches if idx != image_idx]
    else:
        similar_images = [(int(idx), float(distances[idx])) for idx in matches]
    return similar_images
