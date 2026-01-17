# High performance image deduplication by CLIP similarity

## Description

My own CLIP-based image deduplication toolkit born from dissatisfaction with off-the-shelf solutions. (most of them are either slow, or not suitable for processing my own image directories) It's purely command-line, batch processing (not interactive), designed to handle image datasets so large, that typing `ls` inside the directory will take more than 5 seconds for the listing to be done.

Because it's simplicity (less than 1k lines of Python), processing is all done in memory, which limits how much images it can handle in low memory systems. (it takes about 5KiB of VRAM and system RAM for each image)

## Key features

* High performance (as fast as humanly possible on image encoding and matching)
* Incremental embedding DB (filesystem-as-db) with mtime-based updates and orphan cleanup
* Multiple duplicate-keeping strategies (newest, largest, highest-quality, pic-dir, can be extended quite easily)
* GPU support, image embedding comparison is more than 10x faster on GPU (memory-BW bound)
* async batched model inference (about 1.5x speedup) and multiprocessing DB loading (about 2x speedup).

## Installation

Git clone, uv pip install...you know the drill.

```shell
$ git clone https://github.com/NeoChen1024/clip-image-deduper
```

Then install it inside venv, I recommend using uv to manage it (it's going take quite a bit of space because of PyTorch):

```shell
$ cd clip-image-deduper
$ uv venv
$ source .venv/bin/activate
$ uv pip install -e .
```

## Quickstart

It installs the following commands:

* clip-image-deduper: The default deduper implementation, for deduping a image directory with itself.
* clip-image-import-deduper: Alternative deduper implementation, for deduping a "importing" image directory with a "base" dir.
* clip-image-deduper-db-test: Test DB encoding and loading speed
* clip-image-encoding-test: Test a set of images' euclidean distance with each other

Dedupe images in a directory:

```shell
$ clip-image-deduper -i pic-dir -d db-dir -t trash-dir
```

Dedupe images in an "importing" dir with "base" dir (will remove images from "importing" when same image is found in "base"):

```shell
$ clip-image-import-deduper -bi pic-dir -bd pic-db-dir -ii importing -id import-db-dir -t trash-dir
```

## DB Structure & How It Works

The "db" is a directory containing image embeddings that mirrors the image directory structure.
For each image file, there is a corresponding ".npy" file containing the embedding. (`.npy` extension
is added to the original image filename, e.g. `picdir/dir-a/image.jpg` ->    `dbdir/dir-a/image.jpg.npy`)

It uses euclidean distance to calculate similarity (in FP32). (extremely low arithmetic intensity, memory-BW bound)

## Roadmap:

* [ ] Find more ways to save memory
* [ ] Switch to more usable inference library to replace Open CLIP (it has almost no documentations, and gives a ton of linter error)
* [ ] Train custom model to optimize for anime image comparison?
* [ ] Clean-up?

## Current Performance:

Test platform:

Python 3.12 on Arch Linux, AMD Ryzen 7 5700X3D + NVIDIA RTX4080

Image encoding: about 15 image/s

Dedupe: main.py: ~900 image/s for 60k images dataset
