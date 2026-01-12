# Scripts to perform image deduplication by CLIP similarity

This repo contains the following scripts:

* [ ] pic_dir_dedupe.py: dedupe according to my personal image directory structure and image sources.
* [ ] generic_deduper.py: normal dedupe.

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
