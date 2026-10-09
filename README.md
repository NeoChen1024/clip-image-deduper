# Image deduplication by CLIP similarity

My own CLIP-based image deduplication toolkit, born from dissatisfaction with off-the-shelf solutions (most of them are
either slow or not suitable for my own image directories). Purely command-line, batch processing, designed for image
collections so large that `ls` in the directory takes more than 5 seconds.

Everything is held in memory: embeddings are stored and kept on the GPU in FP16, about 2.3 KiB per image for the
default 1152-dimensional model, so a million images need about 2.3 GiB of RAM and VRAM.

## Key features

* Incremental embedding database: one SQLite file per image directory, updated by mtime, orphans cleaned up, bound to
  the model it was encoded with.
* Fast matching: 100k images against each other in about a second on an RTX 4080-class GPU. Queries are processed in
  blocks, so the work is a few hundred large kernels instead of one memory-bound pass per image.
* Exact distances: on CUDA a small Triton kernel reads the FP16 embeddings and computes `sum((a-b)^2)` in FP32
  directly, avoiding the `|a|^2+|b|^2-2ab` cancellation that costs `torch.cdist` about 1e-2 of precision on
  unnormalized CLIP embeddings.
* Configurable keeping policies (which copy of a duplicate group survives), declared in TOML; four built in.
* Image decoding in worker processes, overlapped with model inference.

## Installation

Requirements: Python 3.11+, a working PyTorch install (CUDA strongly recommended), and RAM/VRAM for your collection.

```shell
git clone https://github.com/NeoChen1024/clip-image-deduper
cd clip-image-deduper
uv venv && source .venv/bin/activate
uv pip install -e .
```

A plain `python -m venv .venv` + `pip install -e .` works too. `pip install -e '.[training]'` adds the dependencies of
the CLIP fine-tuning scripts under `src/clip_training`, which are not needed for deduplication.

## Usage

One command with subcommands; `clip-image-deduper <subcommand> --help` lists every option.

Find duplicates within a directory and move the losers to a trash directory:

```shell
clip-image-deduper dedupe -i pictures -d pictures.sqlite --trash-dir trash -k highest-quality
```

Remove images from an "importing" directory that already exist in a "base" collection:

```shell
clip-image-deduper import --base-image-dir pictures --base-db pictures.sqlite \
                          --import-image-dir incoming --import-db incoming.sqlite --trash-dir trash
```

Only (re)encode a directory into its database, or print the distance matrix of a few images to pick a threshold:

```shell
clip-image-deduper update-db -i pictures -d pictures.sqlite
clip-image-deduper encode-test a.jpg a_copy.png b.jpg
```

### Options worth knowing

* `-t, --threshold` (default `0.1`): Euclidean distance at or below which two images are duplicates. With the default
  model, bit-identical re-encodes are well under 0.1; a JPEG q90 re-save, a WebP or a 50% downscale of the same picture
  land roughly between 0.4 and 3, and different pictures at 5 and above. Raise the threshold if you want lossy copies
  caught too; `encode-test` on a few known pairs is the quickest way to calibrate.
* `-k, --keeping-logic` (default `largest`) and `--keeping-config`: see below.
* `-n, --dry-run`: report what would be moved without moving. The database is still refreshed; add `--skip-update`
  for a run that writes nothing at all.
* `-m, --model-id`: any open_clip model. A database is bound to the model it was encoded with; switching models on an
  existing database requires `-f, --force-update`, which re-encodes everything.
* `-b, --batch-size`: images per forward pass; raise it if VRAM allows. Also sets the number of decoder processes.
* Files whose extension Pillow does not recognize are ignored. Files that fail to decode are skipped with a warning and
  retried on the next run. Unlike most libraries' defaults, very large images and truncated files are accepted, like an
  image viewer would.

### Keeping policies

A policy is an ordered list of criteria over the metadata stored in the database (size, mtime, width, height, pixels,
format, and regexes on the path). Duplicates are sorted by the criteria and the first one is kept. Built in, from
`src/clip_image_deduper/policies.toml`:

| name | keeps |
|---|---|
| `newest` | latest mtime, then larger file |
| `largest` | largest file, then newer |
| `highest-quality` | most pixels, then PNG > TIFF > BMP > WebP > AVIF > JPEG > GIF, then larger file, then newer |
| `pic-dir` | files in a `Wallpaper` folder, then by source site in the filename (Pixiv > yande.re > Danbooru > Konachan), then as `highest-quality` |

`pic-dir` encodes the author's own collection layout; it is there as an example of a site-specific policy. Add your
own or override a built-in with `--keeping-config my.toml`:

```toml
[policy.originals-first]
description = "Prefer the raw/ folder, then the biggest file"
criteria = [
    { attr = "dirname", match = '^raw/' },
    { attr = "size", prefer = "max" },
]
```

`policies.toml` documents the three criterion kinds (`prefer`, `match`, `order`).

## How it works

**Database.** Table `embeddings` has one row per image: `path` (relative to the image directory), `mtime`, `size`,
`width`, `height`, `format` as observed at encoding time, and the raw little-endian float16 embedding. Table `meta`
records `model_id`, `dim`, `dtype` and `schema_version`. FP16 storage is lossless: the model runs in FP16 on CUDA, so
every value it emits is already an FP16 number. On update, stored mtimes are compared with the files on disk; changed
or new images are (re)encoded and rows without a file are deleted. Writes are committed per batch, so an interrupted
run keeps its progress.

**Matching.** Embeddings are loaded as one `(N, D)` array and uploaded once. The self-dedupe searches the upper
triangle of the distance matrix in blocks of rows (block size chosen to keep a distance block under 256 MiB), collects
all pairs under the threshold and merges them into connected components, since A~B and B~C does not imply A~C.
On CUDA with Triton (bundled with PyTorch on Linux) the embeddings stay FP16 on the device and `l2_triton.py`
computes exact L2 distances with FP32 arithmetic. Elsewhere they are upcast to FP32 and `torch.cdist` is used; its
matmul form loses about 1e-2 near zero distance, which is why the kernel exists. Peak VRAM for 100k images is about
0.7 GiB on the Triton path and 1.1 GiB on the fallback.

**Model.** The default is `hf-hub:timm/ViT-SO400M-16-SigLIP2-512`. `PE-Core-bigG-14-448` was the previous default
and is noticeably more sensitive to compression artifacts: a JPEG q90 re-save of a picture lands 5 to 15 away from the
original, overlapping with the distance between different pictures.

## Roadmap

* [ ] Store a mean-centered copy or normalized embeddings to make thresholds model-independent?
* [ ] Train a custom model for anime image comparison? (`src/clip_training`)
