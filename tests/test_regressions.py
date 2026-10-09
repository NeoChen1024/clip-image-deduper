import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from clip_image_deduper.dedupe_import_dir import main as import_dedupe_main
from clip_image_deduper.main import find_duplicate_groups, main as self_dedupe_main, select_image_to_keep
from clip_image_deduper.similarity import DistanceIndex, find_similar_images_euclidean
from clip_training.dataset_preprocessing import output_relative_image_path


class _Recorder:
    def __init__(self, items=()):
        self.messages = []
        self.items = list(items)

    def __iter__(self):
        return iter(self.items)

    def write(self, message):
        self.messages.append(message)

    def update(self, n=1):
        pass


class SimilarityRegressionTests(unittest.TestCase):
    def test_find_similar_images_on_slice_does_not_drop_first_match(self):
        embeddings = np.array([[0.0], [0.9], [5.0]], dtype=np.float32)
        matches = find_similar_images_euclidean(-1, embeddings[0], torch.from_numpy(embeddings[1:]), threshold=1.0)
        self.assertEqual(len(matches), 1)
        self.assertEqual([idx for idx, _ in matches], [0])
        self.assertAlmostEqual(matches[0][1], 0.9, places=6)

    def test_duplicate_groups_merge_chain_matches(self):
        image_paths = ["A", "B", "C"]
        embeddings = np.array([[0.0], [0.9], [1.8]], dtype=np.float32)
        groups = find_duplicate_groups(
            image_paths,
            DistanceIndex(embeddings, "cpu"),
            threshold=1.0,
            t=_Recorder(image_paths),
        )
        self.assertEqual(groups, [["A", "B", "C"]])

    def test_duplicate_groups_span_block_boundaries(self):
        # Force tiny query blocks so pairs straddle block edges; each unordered pair must be found exactly once.
        rng = np.random.default_rng(0)
        base = rng.standard_normal((40, 8)).astype(np.float32) * 10
        dup = base[[3, 17, 29, 38]] + 0.01  # four near-duplicates of existing rows
        embeddings = np.concatenate([base, dup]).astype(np.float32)
        image_paths = [f"img{i}" for i in range(len(embeddings))]
        index = DistanceIndex(embeddings, "cpu")
        with mock.patch.object(DistanceIndex, "query_block_size", return_value=7):
            recorder = _Recorder()
            groups = find_duplicate_groups(image_paths, index, threshold=0.5, t=recorder)
        self.assertEqual(sorted(sorted(g) for g in groups), sorted([["img17", "img41"], ["img29", "img42"], ["img3", "img40"], ["img38", "img43"]]))
        self.assertEqual(len(recorder.messages), 4)


class KeepingLogicRegressionTests(unittest.TestCase):
    def test_newest_uses_size_as_tiebreaker(self):
        with tempfile.TemporaryDirectory() as root_dir:
            small_path = os.path.join(root_dir, "small.jpg")
            big_path = os.path.join(root_dir, "big.jpg")
            with open(small_path, "wb") as f:
                f.write(b"x")
            with open(big_path, "wb") as f:
                f.write(b"xxxxxxxxxx")

            shared_mtime = 1_700_000_000
            os.utime(small_path, (shared_mtime, shared_mtime))
            os.utime(big_path, (shared_mtime, shared_mtime))

            self.assertEqual(select_image_to_keep(root_dir, ["small.jpg", "big.jpg"], "newest"), "big.jpg")

    def test_largest_uses_mtime_as_tiebreaker(self):
        with tempfile.TemporaryDirectory() as root_dir:
            old_path = os.path.join(root_dir, "old.jpg")
            new_path = os.path.join(root_dir, "new.jpg")
            with open(old_path, "wb") as f:
                f.write(b"xxxxx")
            with open(new_path, "wb") as f:
                f.write(b"yyyyy")

            os.utime(old_path, (1_700_000_000, 1_700_000_000))
            os.utime(new_path, (1_800_000_000, 1_800_000_000))

            self.assertEqual(select_image_to_keep(root_dir, ["old.jpg", "new.jpg"], "largest"), "new.jpg")


class DatasetPreprocessingRegressionTests(unittest.TestCase):
    def test_output_relative_image_path_preserves_directory_structure(self):
        nested = output_relative_image_path(os.path.join("a", "b.jpg"))
        flat = output_relative_image_path("a_b.jpg")
        self.assertNotEqual(nested, flat)
        self.assertEqual(nested, os.path.join("images", "a", "b.jpg"))


class DryRunRegressionTests(unittest.TestCase):
    def test_self_dedupe_dry_run_still_refreshes_database(self):
        with mock.patch("clip_image_deduper.main.CLIPImageEncoder") as encoder_cls, mock.patch(
            "clip_image_deduper.main.update_database"
        ) as update_database, mock.patch("clip_image_deduper.main.load_database") as load_database, mock.patch(
            "clip_image_deduper.main.find_duplicate_groups"
        ) as find_duplicate_groups:
            encoder = mock.Mock()
            encoder_cls.return_value = encoder
            load_database.return_value = (["a.jpg"], np.array([[0.0]], dtype=np.float32))
            find_duplicate_groups.return_value = []

            self_dedupe_main.callback(
                image_dir="images",
                db="db.sqlite",
                model_id="model",
                force_update=False,
                clean_orphans=True,
                device="cpu",
                skip_update=False,
                dry_run=True,
                threshold=0.1,
                trash_dir=None,
                keeping_logic="largest",
                batch_size=4,
            )

            encoder_cls.assert_called_once_with(model_id="model", device="cpu")
            update_database.assert_called_once_with(encoder, "images", "db.sqlite", False, True, batch_size=4)

    def test_import_dedupe_dry_run_still_refreshes_both_databases(self):
        with mock.patch("clip_image_deduper.dedupe_import_dir.CLIPImageEncoder") as encoder_cls, mock.patch(
            "clip_image_deduper.dedupe_import_dir.update_database"
        ) as update_database, mock.patch("clip_image_deduper.dedupe_import_dir.load_database") as load_database:
            encoder = mock.Mock()
            encoder_cls.return_value = encoder
            load_database.side_effect = [
                (["base.jpg"], np.array([[0.0]], dtype=np.float32)),
                (["import.jpg"], np.array([[0.0]], dtype=np.float32)),
            ]
            import_dedupe_main.callback(
                base_image_dir="base-images",
                base_db="base.sqlite",
                import_image_dir="import-images",
                import_db="import.sqlite",
                model_id="model",
                force_update=False,
                clean_orphans=True,
                device="cpu",
                skip_update=False,
                dry_run=True,
                threshold=0.1,
                trash_dir=None,
                batch_size=4,
            )

            encoder_cls.assert_called_once_with(model_id="model", device="cpu")
            self.assertEqual(update_database.call_count, 2)
            update_database.assert_has_calls(
                [
                    mock.call(encoder, "base-images", "base.sqlite", False, True, batch_size=4),
                    mock.call(encoder, "import-images", "import.sqlite", False, True, batch_size=4),
                ]
            )


if __name__ == "__main__":
    unittest.main()
