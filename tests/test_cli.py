import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from click.testing import CliRunner

from clip_image_deduper import cli as cli_mod
from clip_image_deduper.cli import cli
from clip_image_deduper.db_store import ImageRecord


def _records(*names):
    return [ImageRecord(n, 0.0, 1, 1, 1, "JPEG") for n in names]


class CliTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.tmp = tempfile.TemporaryDirectory()
        self.img = os.path.join(self.tmp.name, "img")
        os.makedirs(self.img)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dedupe_dry_run_still_refreshes_database(self):
        with mock.patch.object(cli_mod, "CLIPImageEncoder") as enc_cls, mock.patch.object(cli_mod, "update_database") as upd, mock.patch.object(
            cli_mod, "load_database", return_value=(_records("a.jpg", "b.jpg"), np.array([[0.0], [0.05]], dtype=np.float32))
        ):
            result = self.runner.invoke(cli, ["dedupe", "-i", self.img, "-d", "db.sqlite", "-n", "--trash-dir", "trash", "-c", "cpu", "-m", "model"])
        self.assertEqual(result.exit_code, 0, result.output)
        enc_cls.assert_called_once_with(model_id="model", device="cpu", dtype=None)
        enc_cls.return_value.close.assert_called_once()
        upd.assert_called_once()
        self.assertEqual(upd.call_args.args[1:3], (self.img, "db.sqlite"))
        self.assertTrue(upd.call_args.kwargs["clean_orphans"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "trash")))  # dry run moved nothing

    def test_dedupe_skip_update_does_not_load_model(self):
        with mock.patch.object(cli_mod, "CLIPImageEncoder") as enc_cls, mock.patch.object(
            cli_mod, "load_database", return_value=(_records("a.jpg"), np.array([[0.0]], dtype=np.float32))
        ):
            result = self.runner.invoke(cli, ["dedupe", "-i", self.img, "-d", "db.sqlite", "--skip-update", "-c", "cpu"])
        self.assertEqual(result.exit_code, 0, result.output)
        enc_cls.assert_not_called()

    def test_unknown_policy_fails_before_encoding(self):
        with mock.patch.object(cli_mod, "CLIPImageEncoder") as enc_cls:
            result = self.runner.invoke(cli, ["dedupe", "-i", self.img, "-d", "db.sqlite", "-k", "nope"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Unknown keeping policy", result.output)
        enc_cls.assert_not_called()

    def test_import_dry_run_refreshes_both_databases(self):
        base_img = os.path.join(self.tmp.name, "base")
        os.makedirs(base_img)
        with mock.patch.object(cli_mod, "CLIPImageEncoder"), mock.patch.object(cli_mod, "update_database") as upd, mock.patch.object(
            cli_mod, "load_database", side_effect=[(_records("b.jpg"), np.array([[0.0]], dtype=np.float32)), (_records("i.jpg"), np.array([[0.0]], dtype=np.float32))]
        ):
            result = self.runner.invoke(
                cli, ["import", "--base-image-dir", base_img, "--base-db", "b.sqlite", "--import-image-dir", self.img, "--import-db", "i.sqlite", "-n", "-c", "cpu"]
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual([c.args[1:3] for c in upd.call_args_list], [(base_img, "b.sqlite"), (self.img, "i.sqlite")])

    def test_empty_database_is_an_error(self):
        with mock.patch.object(cli_mod, "load_database", return_value=([], np.empty((0, 0)))):
            result = self.runner.invoke(cli, ["dedupe", "-i", self.img, "-d", "db.sqlite", "--skip-update", "-c", "cpu"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No entries", result.output)


if __name__ == "__main__":
    unittest.main()
