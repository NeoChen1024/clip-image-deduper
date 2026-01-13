#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Any, Dict, List, Tuple

import numpy as np

default_cosine_similarity_threshold = 0.97
default_euclidean_distance_threshold = 0.5


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalized cosine similarity between two sets of vectors."""
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a_norm @ b_norm.T


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the Euclidean distance matrix between two sets of vectors.

    Accepts a, b as (D,), (1, D), or (N, D)/(M, D).
    Returns shape (N, M).
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=-1)


def find_similar_images_cosine(
    image_idx: int, image_embedding: np.ndarray, database: np.ndarray, threshold: float = default_cosine_similarity_threshold
) -> List[Tuple[int, float]]:
    """Find similar images in the database based on cosine similarity."""
    # print(f"dimensions: i = {image_embedding.shape}, d = {database.shape}")
    similarities = cosine_similarity(image_embedding, database)
    matches = np.where(similarities >= threshold)
    similar_images = [(int(m), float(similarities[m])) for m in matches[0]]
    # remove self-match
    similar_images = [(idx, sim) for idx, sim in similar_images if idx != image_idx]
    return similar_images


def find_similar_images_euclidean(
    image_idx: int, image_embedding: np.ndarray, database: np.ndarray, threshold: float = default_euclidean_distance_threshold
) -> List[Tuple[int, float]]:
    """Find similar images in the database based on Euclidean distance."""
    distances = euclidean_distance(image_embedding, database)

    # For a single query vector vs database, distances has shape (1, N).
    # Squeeze to 1D so indexing and thresholding behave as expected.
    if distances.ndim == 2 and distances.shape[0] == 1:
        distances = distances[0]

    matches = np.where(distances <= threshold)[0]
    similar_images = [(int(idx), float(distances[idx])) for idx in matches if idx != image_idx]
    return similar_images
