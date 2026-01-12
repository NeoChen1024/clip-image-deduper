# Scripts to perform image deduplication by CLIP similarity

This repo contains the following scripts:

* [X] update_db.py: Update embedding when files' mtime changed, or embedding isn't present now.
* [ ] dedupe_by_embedding.py: Do the actual dedupe process.
* [ ] pic_dir_dedupe.py: dedupe according to my personal image directory structure and image sources.

# "db" structure

The "db" is a directory containing image embeddings that mirrors the image directory structure.
For each image file, there is a corresponding ".npy" file containing the embedding. (`.npy` extension
is added to the original image filename, e.g. `image.jpg` -> `image.jpg.npy`)

# Roadmap:

* [X] Support updating embedding when image files are changed.
* [X] Support cleaning up db entries when image files are deleted.
* [ ] Implement deduplication logic.
* [ ] Batched embedding computation for better performance. (even better: async + multiprocessing)
* [ ] Parallelized deduplication process.
