#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from typing import List, Dict
import tqdm
import click
import math
import PIL.Image
import numpy as np
import torch
from .db_processing import update_database
from .clip_encoding import CLIPImageEncoder, default_model_id


@click.command()
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing images to process.",
)
@click.option(
    "--db-dir",
    "-d",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Directory to store the database JSON files.",
)
@click.option("--force-update", "-f", is_flag=True, default=False, help="Force update all images, ignoring modification times.")
@click.option("--clip-model", "-m", default=default_model_id, help="CLIP model to use for encoding images.", show_default=True)
@click.option("--device", "-c", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the CLIP model on.", show_default=True)
def main(image_dir: str, db_dir: str, force_update: bool, clip_model: str, device: str):
    print(f"Using CLIP model: {clip_model}, on device {device}")
    encoder = CLIPImageEncoder(model_id=clip_model, device=device)

    # don't need to catch exceptions here, they are handled in update_database
    def process_image(image_path: str, db_npz_path: str):
        with PIL.Image.open(image_path) as img:
            img = img.convert("RGB")
            encoding = encoder.encode_image(img)

        # save embedding as NumPy npz file
        np.savez_compressed(db_npz_path, clip_embedding=encoding)

    update_database(image_dir, db_dir, process_image, data_extension=".npz", force_update=force_update)


if __name__ == "__main__":
    main()
