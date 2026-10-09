# High performance image deduplication by CLIP similarity

## Description

My own CLIP-based image deduplication toolkit born from dissatisfaction with off-the-shelf solutions. (most of them are either slow, or not suitable for processing my own image directories) It's purely command-line, batch processing (not interactive), designed to handle image datasets so large, that typing `ls` inside the directory will take more than 5 seconds for the listing to be done.

Because of its simplicity (about 1k lines of Python), processing is all done in memory, which limits how many images it can handle in low-memory systems. Embeddings are stored and held on the GPU in FP16, so each image costs about 2.5KiB of VRAM and system RAM for a 1152-dimensional embedding; a 1M-image database fits in about 2.5GiB.

## Key features

* High performance (as fast as humanly possible on image encoding and matching)
* Incremental embedding DB (a single SQLite file, FP16) with mtime-based updates, orphan cleanup, and model-id checking
* Multiple duplicate-keeping strategies (newest, largest, highest-quality, pic-dir, can be extended quite easily)
* GPU support: matching 100k images against each other takes about a second on an RTX 4080-class GPU
* Exact distances: on CUDA a small Triton kernel reads FP16 embeddings and computes `sum((a-b)^2)` in FP32 directly, avoiding the `|a|^2+|b|^2-2ab` cancellation that costs `torch.cdist` ~1e-2 of precision on unnormalized CLIP embeddings
* async batched model inference (about 1.5x speedup)

## Installation

Minimum requirements:

* Python 3.11+
* More than 8GiB of free RAM
* A working PyTorch install (CUDA optional but recommended if you have a GPU)

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

If you don't use uv, a plain virtualenv + pip flow also works:

```shell
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -e .
```

## Quickstart

It installs the following commands:

* clip-image-deduper: The default deduper implementation, for deduping a image directory with itself.
* clip-image-import-deduper: Alternative deduper implementation, for deduping a "importing" image directory with a "base" dir.
* clip-image-deduper-db-test: Test DB encoding and loading speed
* clip-image-encoding-test: Test a set of images' euclidean distance with each other

Dedupe images in a directory:

```shell
$ clip-image-deduper -i pic-dir -d pic.sqlite -t trash-dir
```

Dedupe images in an "importing" dir with "base" dir (will remove images from "importing" when same image is found in "base"):

```shell
$ clip-image-import-deduper -bi pic-dir -bd pic.sqlite -ii importing -id import.sqlite -t trash-dir
```

### Important CLI options (clip-image-deduper)

Only the most important flags are listed here; run `clip-image-deduper --help` for the full reference.

* `-i, --image-dir`: Directory containing images to process.
* `-d, --db`: SQLite file storing the embedding database (created if missing). A database is bound to the model it was encoded with; switching `--model-id` requires `--force-update`.
* `-t, --trash-dir`: Where duplicates are moved. If omitted, files are not moved.
* `--threshold, -th`: Euclidean distance threshold for considering images as duplicates. Default: `0.1` (lower = stricter).
* `--keeping-logic, -kl`: Which copy to keep among duplicates: `newest`, `largest`, `highest-quality`, or `pic-dir`.
* `--device, -c`: Device to run the CLIP model on, e.g. `cuda` or `cpu`. Defaults to `cuda` if available.
* `--batch-size, -b`: Batch size for image encoding. Adjust based on VRAM.
* `--dry-run, -n`: Show what would be moved without moving image files. The embedding DB is still refreshed unless you also pass `--skip-update`.

## DB Structure & How It Works

The "db" is a single SQLite file. Table `embeddings` has one row per image: `path` (relative to the image dir, primary key), `mtime` and `size` of the image when it was encoded, and `embedding`, the raw little-endian float16 vector as a BLOB. Table `meta` records `model_id`, `dim`, `dtype` and `schema_version`. FP16 is lossless here: CLIP models run in FP16 on CUDA, so every output value is already exactly representable.

On update, the stored mtimes are compared with the files on disk: images with a changed mtime or no row get (re)encoded, rows whose image no longer exists are deleted (`--clean-orphans`). Loading concatenates all BLOBs and reinterprets them as one `(N, D)` float16 array with no per-row parsing. Writes are committed per encoding batch, so an interrupted run keeps its progress.

### Matching

Matching is done in blocks of query rows against the (upper triangle of the) whole database, so it is a few hundred large kernel launches instead of one memory-bound pass per image. Block size is chosen so a distance block stays under 256MiB.

* On CUDA with Triton available (it ships with PyTorch on Linux), the database lives on the GPU as FP16 and `l2_triton.py` computes exact L2 distances with FP32 arithmetic from the FP16 values, i.e. no `|a|^2+|b|^2-2ab` cancellation. Error vs. float64 is ~1e-5.
* Otherwise (CPU, or CUDA without Triton) the database is upcast to FP32 and `torch.cdist` is used. Its matmul form loses ~1e-2 of absolute precision near zero distance for embeddings of norm ~16, which has not been observed to change any decision at the default threshold, but is why the Triton path exists.

Peak VRAM for 100k images is about 0.7GiB on the Triton path and 1.1GiB on the `torch.cdist` path.


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
