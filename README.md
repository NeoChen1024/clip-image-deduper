# Scripts to perform image deduplication by CLIP similarity

This repo contains the following scripts:

* update_db.py: Update embedding when files' mtime / checksum changed, or embedding isn't present now.
* dedupe_by_embedding.py: Do the actual dedupe process.
* pic_dir_dedupe.py: dedupe according to my personal image directory structure and image sources.

``
