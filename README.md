# Scripts to perform image deduplication by CLIP similarity

This repo contains the following scripts:

* [X] main.py: normal dedupe.
* [X] dedupe_import_dir.py: remove duplicates from importing dir.

# "db" structure

The "db" is a directory containing image embeddings that mirrors the image directory structure.
For each image file, there is a corresponding ".npy" file containing the embedding. (`.npy` extension
is added to the original image filename, e.g. `image.jpg` -> `image.jpg.npy`)

# Roadmap:

* [X] Support updating embedding when image files are changed.
* [X] Support cleaning up db entries when image files are deleted.
* [X] Implement deduplication logic.
* [ ] Async batched embedding computation for better performance.
