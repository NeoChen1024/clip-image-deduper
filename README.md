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
* Three-stage pipeline: decoder processes read and fully preprocess images, the main thread gathers batches, and
  the GPU runs one batch while the next is being decoded. Decoder count and batch size are independent knobs.

## Installation

Requirements: Python 3.11+, a working PyTorch install (CUDA strongly recommended), and RAM/VRAM for your collection.
The default model is downloaded from the Hugging Face Hub on first use (1.6 GiB).

```shell
git clone https://github.com/NeoChen1024/clip-image-deduper
cd clip-image-deduper
uv venv && source .venv/bin/activate
uv pip install -e .
```

A plain `python -m venv .venv` + `pip install -e .` works too. `pip install -e '.[training]'` adds the dependencies of
the CLIP fine-tuning scripts under `src/clip_training` (see `docs/training-clip.md`), which are not needed for
deduplication.

## Usage

One command with subcommands; `clip-image-deduper <subcommand> --help` lists every option.

Find duplicates within a directory and move the losers to a trash directory:

```shell
clip-image-deduper dedupe -i pictures -d pictures.sqlite --trash-dir trash -k highest-quality
```

Remove images from an "importing" directory that already exist in a "base" collection:

```shell
clip-image-deduper dedupe-import --base-image-dir pictures --base-db pictures.sqlite \
                                 --import-image-dir incoming --import-db incoming.sqlite --trash-dir trash
```

Only (re)encode a directory into its database, or print the distance matrix of a few images to pick a threshold:

```shell
clip-image-deduper update-db -i pictures -d pictures.sqlite
clip-image-deduper encode-test a.jpg a_copy.png b.jpg
```

Calibrate a threshold from the collection itself: sample images, synthesize lossy copies of them, and print
histograms of how far the copies land from the stored embedding next to how far the nearest *different* image is:

```shell
clip-image-deduper calibrate -i pictures -d pictures.sqlite -s 200 --seed 0 --variants jpeg90,half,quarter
```

### Options worth knowing

* `-t, --threshold` (default `0.1`): Euclidean distance at or below which two images are duplicates. With the default
  model, bit-identical re-encodes are well under 0.1; a JPEG q90 re-save or a 50% downscale of the same picture lands
  around 0.2-0.5, a 25% downscale up to about 2.4, and different pictures at 5.5 and above. Raise the threshold if
  you want lossy copies caught too; `calibrate` measures these distributions on your own collection and model and
  suggests a value, `encode-test` does the same for a handful of files by hand.
* `calibrate` options: `-s, --samples` (default 200) images are drawn with `--seed` (default 0), so an unchanged
  directory gives the same sample again (the draw is over sorted paths, independent of database row order; it is not
  promised to survive version upgrades). `--variants` picks which lossy copies to synthesize (`jpeg90`, `jpeg75`,
  `webp80`, `half`, `quarter`), `-t` may be repeated to see what several thresholds would catch. The suggestion is
  the largest variant distance rounded up, provided it is below the nearest different image; the default 0.1 is
  intentionally far below it and is not what this command recommends.
* `-k, --keeping-logic` (default `largest`) and `--keeping-config`: see below.
* `-n, --dry-run`: report what would be moved without moving. The database is still refreshed; add `--skip-update`
  for a run that writes nothing at all.
* `-m, --model-id`: a timm model name; any CLIP/SigLIP image tower timm ships works, NaFlex (`naflexvit_*`) or
  fixed-resolution (e.g. `vit_so400m_patch16_siglip_512.v2_webli`, `vit_pe_core_bigG_14_448.fb`). A database is bound
  to the model it was encoded with; switching models on an existing database requires `-f, --force-update`, which
  re-encodes everything.
* `--seq-len` (default `1024`): for NaFlex towers, the patch budget per image. The image keeps its aspect ratio and is
  resized to at most this many 16x16 patches; 1024 is the pixel budget of a 512px fixed model and costs the same
  time and VRAM. Lower it on small GPUs. It is part of the identity a database is bound to.
* `-b, --batch-size`: images per forward pass; raise it if VRAM allows. The default model is compute bound already at
  4, so this barely changes throughput.
* `-j, --workers`: decoder processes (default: one per CPU); twice as many reads are kept in flight. On network or
  spinning storage the reads are the bottleneck, so a value above the CPU count helps there.
* `--compile`: `torch.compile` the model with dynamic shapes, so the batch size stays free. About 17% faster for
  fixed-resolution towers, 20-30 s warm-up per run. No gain for the default NaFlex tower yet: timm's NaFlex
  position-embedding code breaks the graph.
* FP16 inference is not batch-invariant: the same image encoded in a different batch (size or neighbours) comes out
  up to ~0.04 away, with or without `--compile`. That is far below the default threshold and the ~0.4+ of a lossy
  copy, but it is the reason distances between near-identical files are not exactly zero.
* Files whose extension Pillow does not recognize are ignored. Files that fail to decode are skipped with a warning and
  retried on the next run. Unlike Pillow's defaults, very large images, truncated files and PNGs with bad checksums on
  metadata chunks (e.g. the `iCCP` chunk some Pixiv uploads carry) are accepted, like an image viewer would.

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

**Encoding pipeline.** `--workers` processes each open a file, decode it, and run the model's full preprocessing
(resize, normalize, patchify, cast to the model dtype), so the main process only ever sees ready-to-stack arrays. It keeps
`2 * workers` decode jobs in flight, gathers results into batches as they complete (a slow file never blocks the
others), stages each batch in pinned memory and submits it to the GPU without waiting; the previous batch's result is
collected and written to the database only once the next one is queued. Memory is bounded by `2 * workers` arrays
plus two batches. Measured on 1000 cached 4 MB images (RTX 4080, 16 CPUs): the GPU alone does 65 img/s at any batch
size, 16 decoders alone 74 img/s, and the pipeline reaches 42 img/s against 24 img/s for the previous version at the
same batch size of 4.

**Model.** Only the image tower is needed, so it is loaded straight from timm (`naflexvit_so400m_patch16_siglip.v2_webli`,
the SigLIP2 so400m NaFlex tower, 1.6 GiB; the full checkpoint with the text tower is 4.3 GiB). NaFlex takes the image
at its own aspect ratio: the decoder resizes it to at most `--seq-len` 16x16 patches, normalizes and patchifies it,
and the encoder pads the patch sequences of a batch to `--seq-len` with a validity mask. Nothing is squashed or
cropped, which is what makes it tighter than the fixed-size towers on the variants a deduper cares about. Measured on
24 library images against `vit_so400m_patch16_siglip_512` (the previous default), as distance to the original /
smallest distance between different images: JPEG q90 0.5 / 5.6 vs 0.7 / 8.7, 50% downscale 0.4 / 5.6 vs 1.4 / 8.7,
25% downscale 2.4 / 5.6 vs 4.3 / 8.7. Same speed and VRAM at `--seq-len 1024`. Fixed-resolution towers still work:
for them the encoder forces `crop_pct=1.0, crop_mode="squash"` with bicubic resampling, which is what they were
trained with (timm's default eval transform center-crops 90% and shifts embeddings by several units).
`PE-Core-bigG-14-448` was the default before SigLIP2 and is noticeably more sensitive to compression artifacts: a
JPEG q90 re-save of a picture lands 5 to 15 away from the original, overlapping with different pictures.

## Roadmap

* [ ] Store a mean-centered copy or normalized embeddings to make thresholds model-independent? (`calibrate` now
  measures the scale per model instead.)
* [ ] PySide6 review GUI for the band between the automatic and a looser threshold: plan in
  [docs/gui-review.md](docs/gui-review.md).
* [ ] Train a custom model for anime image comparison, and later anime semantic search? Notes in
  [docs/training-clip.md](docs/training-clip.md), scripts under `src/clip_training`.
