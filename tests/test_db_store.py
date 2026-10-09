import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from clip_image_deduper import db_processing
from clip_image_deduper.db_store import EmbeddingDB, ModelMismatchError


class EmbeddingDBTests(unittest.TestCase):
    def test_roundtrip_upsert_load_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "db.sqlite")
            vecs = np.arange(12, dtype=np.float32).reshape(3, 4)
            with EmbeddingDB(path) as db:
                db.ensure_model("model-a")
                db.upsert([(f"{i}.jpg", 100.0 + i, 10 * i, vecs[i]) for i in range(3)])
                self.assertEqual(db.dim, 4)
                self.assertEqual(len(db), 3)
                self.assertEqual(db.load_index(), {"0.jpg": 100.0, "1.jpg": 101.0, "2.jpg": 102.0})

                # replace one row, delete another
                db.upsert([("1.jpg", 999.0, 1, vecs[1] * 2)])
                self.assertEqual(db.delete(["2.jpg"]), 1)

            with EmbeddingDB(path) as db:
                paths, arr = db.load_all()
                self.assertEqual(paths, ["0.jpg", "1.jpg"])
                self.assertEqual(arr.dtype, np.float16)
                self.assertEqual(arr.shape, (2, 4))
                np.testing.assert_array_equal(arr[0], vecs[0])
                np.testing.assert_array_equal(arr[1], vecs[1] * 2)
                self.assertTrue(arr.flags.writeable)

    def test_model_mismatch_raises_unless_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "db.sqlite")
            with EmbeddingDB(path) as db:
                db.ensure_model("model-a")
                db.upsert([("a.jpg", 1.0, 1, np.zeros(2, dtype=np.float32))])
            with EmbeddingDB(path) as db:
                with self.assertRaises(ModelMismatchError):
                    db.ensure_model("model-b")
                db.ensure_model("model-b", reset_on_mismatch=True)
                self.assertEqual(len(db), 0)
                self.assertEqual(db.model_id, "model-b")
                self.assertIsNone(db.dim)

    def test_dimension_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with EmbeddingDB(os.path.join(tmp, "db.sqlite")) as db:
                db.ensure_model("m")
                db.upsert([("a.jpg", 1.0, 1, np.zeros(3, dtype=np.float32))])
                with self.assertRaises(ValueError):
                    db.upsert([("b.jpg", 1.0, 1, np.zeros(4, dtype=np.float32))])

    def test_load_database_missing_file_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, arr = db_processing.load_database(os.path.join(tmp, "missing.sqlite"))
            self.assertEqual(paths, [])
            self.assertEqual(arr.shape[0], 0)


class UpdateDatabaseTests(unittest.TestCase):
    """Drive update_database with a fake encoder; image loading/encoding is stubbed out."""

    def _run_update(self, image_dir, db_path, encoder, **kwargs):
        def fake_encode(encoder, db, image_dir, batch_paths, image_np_batch):
            db.upsert([(p, os.path.getmtime(os.path.join(image_dir, p)), 1, np.array([len(p)], dtype=np.float32)) for p in batch_paths])

        def fake_candidates(encoder, db, image_dir, candidates, batch_size):
            fake_encode(encoder, db, image_dir, candidates, None)

        with mock.patch.object(db_processing, "_encode_candidates", side_effect=fake_candidates) as m:
            db_processing.update_database(encoder, image_dir, db_path, **kwargs)
            return [sorted(call.args[3]) for call in m.call_args_list]

    def test_incremental_update_and_orphan_cleanup(self):
        encoder = mock.Mock(model_id="m")
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = os.path.join(tmp, "img")
            os.makedirs(os.path.join(image_dir, "sub"))
            for rel in ("a.jpg", "sub/b.jpg"):
                with open(os.path.join(image_dir, rel), "wb") as f:
                    f.write(b"x")
                os.utime(os.path.join(image_dir, rel), (1_000, 1_000))
            db_path = os.path.join(tmp, "db.sqlite")

            # first run encodes everything
            self.assertEqual(self._run_update(image_dir, db_path, encoder), [["a.jpg", "sub/b.jpg"]])
            # second run: nothing changed, nothing encoded
            self.assertEqual(self._run_update(image_dir, db_path, encoder), [])
            # touch one, remove the other -> one re-encode, one orphan deleted
            os.utime(os.path.join(image_dir, "a.jpg"), (2_000, 2_000))
            os.remove(os.path.join(image_dir, "sub", "b.jpg"))
            self.assertEqual(self._run_update(image_dir, db_path, encoder), [["a.jpg"]])
            paths, _ = db_processing.load_database(db_path)
            self.assertEqual(paths, ["a.jpg"])
            # force update re-encodes regardless of mtime
            self.assertEqual(self._run_update(image_dir, db_path, encoder, force_update=True), [["a.jpg"]])

    def test_model_mismatch_surfaces_as_click_error(self):
        import click

        with tempfile.TemporaryDirectory() as tmp:
            image_dir = os.path.join(tmp, "img")
            os.makedirs(image_dir)
            db_path = os.path.join(tmp, "db.sqlite")
            self._run_update(image_dir, db_path, mock.Mock(model_id="m1"))
            with self.assertRaises(click.ClickException):
                self._run_update(image_dir, db_path, mock.Mock(model_id="m2"))


if __name__ == "__main__":
    unittest.main()
