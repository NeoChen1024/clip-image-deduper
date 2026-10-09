import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from click.testing import CliRunner

from clip_image_deduper import cli as cli_mod
from clip_image_deduper import review as rv
from clip_image_deduper.cli import cli
from clip_image_deduper.db_store import ImageRecord
from clip_image_deduper.dedupe import group_edges
from clip_image_deduper.keeping import builtin_policies


def rec(path, size=1, w=1, h=1, mtime=0.0):
    return ImageRecord(path, mtime, size, w, h, "PNG")


LARGEST = builtin_policies()["largest"]


class GroupEdgesTests(unittest.TestCase):
    def test_groups_keep_their_edges_and_are_sorted(self):
        groups = group_edges(6, [(4, 5, 0.3), (0, 1, 0.1), (1, 2, 0.2)])
        self.assertEqual([g.members for g in groups], [[0, 1, 2], [4, 5]])
        self.assertEqual(groups[0].edges, [(0, 1, 0.1), (1, 2, 0.2)])
        self.assertEqual((groups[0].min_distance, groups[0].max_distance), (0.1, 0.2))


class ReviewDBTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = rv.ReviewDB(os.path.join(self.tmp.name, "x.review.sqlite"))
        self.recs = {p: rec(p, size=s) for p, s in [("a.png", 10), ("b.png", 20), ("c.png", 5), ("d.png", 1), ("e.png", 2)]}

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def groups(self, *specs):
        return [([self.recs[p] for p in members], edges) for members, edges in specs]

    def test_default_path(self):
        self.assertEqual(rv.default_review_path("/x/pictures.sqlite"), "/x/pictures.review.sqlite")

    def test_upsert_new_predecided_kept_stale(self):
        stats = self.db.upsert_groups(self.groups((["a.png", "b.png"], [("a.png", "b.png", 0.05)]), (["c.png", "d.png"], [("c.png", "d.png", 0.4)])), LARGEST, 0.1)
        self.assertEqual((stats.new, stats.predecided, stats.kept, stats.stale), (2, 1, 0, 0))
        g1, g2 = self.db.groups()
        self.assertEqual((g1.status, g1.n, g1.min_distance), ("decided", 2, 0.05))
        self.assertEqual(g2.status, "pending")
        keeps = {m.path: m.keep for m in self.db.members(g1.id)}
        self.assertEqual(keeps, {"a.png": False, "b.png": True})  # largest wins
        self.assertEqual(self.db.edges(g2.id), [("c.png", "d.png", 0.4)])
        # the reviewer changes group 2, then a re-match keeps group 2 as is and replaces group 1 by a bigger one
        self.db.keep_only(g2.id, "d.png")
        stats = self.db.upsert_groups(
            self.groups((["a.png", "b.png", "e.png"], [("a.png", "b.png", 0.05), ("b.png", "e.png", 0.3)]), (["c.png", "d.png"], [("c.png", "d.png", 0.41)])), LARGEST, 0.1
        )
        self.assertEqual((stats.new, stats.kept, stats.stale), (1, 1, 1))
        self.assertEqual(self.db.counts()["stale"], 1)
        g2b = self.db.group(g2.id)
        self.assertEqual((g2b.status, g2b.min_distance), ("decided", 0.41))
        self.assertEqual({m.path: m.keep for m in self.db.members(g2.id)}, {"c.png": False, "d.png": True})
        self.assertNotIn(g1.id, [g.id for g in self.db.groups()])
        self.assertEqual(self.db.groups_of("b.png"), [g1.id, [g.id for g in self.db.groups("pending")][0]])

    def test_decide_undo_and_status_checks(self):
        self.db.upsert_groups(self.groups((["a.png", "b.png"], [("a.png", "b.png", 0.4)])), LARGEST, 0.1)
        gid = self.db.groups()[0].id
        self.assertFalse(self.db.undo(gid))  # only the initial snapshot
        self.db.decide(gid, keeps={"a.png": True}, note="both are fine")
        self.db.decide(gid, status="skipped")
        self.assertEqual(self.db.group(gid).status, "skipped")
        self.assertTrue(self.db.undo(gid))
        g = self.db.group(gid)
        self.assertEqual((g.status, g.note), ("pending", "both are fine"))
        self.assertEqual({m.path: m.keep for m in self.db.members(gid)}, {"a.png": True, "b.png": True})
        self.assertTrue(self.db.undo(gid))
        self.assertEqual({m.path: m.keep for m in self.db.members(gid)}, {"a.png": False, "b.png": True})
        self.assertFalse(self.db.undo(gid))
        with self.assertRaises(ValueError):
            self.db.decide(gid, status="bogus")
        with self.assertRaises(KeyError):
            self.db.decide(gid, keeps={"zzz": True})
        self.db.reset_to_policy(gid, builtin_policies()["smallest"]) if "smallest" in builtin_policies() else None

    def test_plan_apply_skips_groups_keeping_nothing(self):
        self.db.upsert_groups(self.groups((["a.png", "b.png"], [("a.png", "b.png", 0.05)]), (["c.png", "d.png"], [("c.png", "d.png", 0.05)])), LARGEST, 0.1)
        g1, g2 = self.db.groups()
        self.db.decide(g2.id, keeps={"c.png": False, "d.png": False})
        moves = self.db.plan_apply()
        self.assertEqual([(m.group_id, m.path, m.keep) for m in moves], [(g1.id, "a.png", ("b.png",))])


class ApplyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.img = os.path.join(self.tmp.name, "img")
        os.makedirs(os.path.join(self.img, "sub"))
        self.recs = []
        for p, data in [("a.png", b"aaaa"), ("sub/b.png", b"bb"), ("c.png", b"cccccc"), ("d.png", b"d")]:
            full = os.path.join(self.img, p)
            with open(full, "wb") as f:
                f.write(data)
            st = os.stat(full)
            self.recs.append(ImageRecord(p, st.st_mtime, st.st_size, 1, 1, "PNG"))
        self.db = rv.ReviewDB(os.path.join(self.tmp.name, "x.review.sqlite"))
        self.trash = os.path.join(self.tmp.name, "trash")

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_apply_moves_losers_refuses_changed_and_unapplies(self):
        r = {x.path: x for x in self.recs}
        self.db.upsert_groups(
            [([r["a.png"], r["sub/b.png"]], [("a.png", "sub/b.png", 0.05)]), ([r["c.png"], r["d.png"]], [("c.png", "d.png", 0.05)])], LARGEST, 0.1
        )
        g1, g2 = self.db.groups()
        with open(os.path.join(self.img, "d.png"), "ab") as f:
            f.write(b"changed")
        moved, refused = rv.apply_decisions(self.db, self.img, self.trash, dry_run=True)
        self.assertEqual((moved, refused), (1, [g2.id]))
        self.assertTrue(os.path.exists(os.path.join(self.img, "sub", "b.png")))
        moved, refused = rv.apply_decisions(self.db, self.img, self.trash, dry_run=False)
        self.assertEqual((moved, refused), (1, [g2.id]))
        self.assertFalse(os.path.exists(os.path.join(self.img, "sub", "b.png")))
        self.assertTrue(os.path.exists(os.path.join(self.trash, "sub", "b.png")))
        self.assertEqual(self.db.group(g1.id).status, "applied")
        self.assertEqual(self.db.group(g2.id).status, "decided")
        with self.assertRaises(ValueError):
            self.db.decide(g1.id, status="pending")
        self.assertEqual(rv.unapply_group(self.db, self.img, g1.id), 1)
        self.assertTrue(os.path.exists(os.path.join(self.img, "sub", "b.png")))
        self.assertEqual(self.db.group(g1.id).status, "decided")
        self.assertEqual(self.db.applied(), [])


class CliReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.img = os.path.join(self.tmp.name, "img")
        os.makedirs(self.img)
        self.db = os.path.join(self.tmp.name, "pictures.sqlite")
        self.records = [ImageRecord(p, 0.0, s, 1, 1, "PNG") for p, s in [("a.png", 1), ("b.png", 2), ("c.png", 3), ("d.png", 4)]]
        self.emb = np.array([[0.0], [0.05], [0.3], [5.0]], dtype=np.float32)

    def tearDown(self):
        self.tmp.cleanup()

    def run_dedupe(self, *extra):
        with mock.patch.object(cli_mod, "load_database", return_value=(self.records, self.emb)), mock.patch.object(cli_mod, "EmbeddingDB") as store:
            store.return_value.__enter__.return_value.model_id = "m"
            return CliRunner().invoke(cli, ["dedupe", "-i", self.img, "-d", self.db, "--skip-update", "-c", "cpu", *extra])

    def test_review_threshold_stores_groups_and_moves_nothing(self):
        res = self.run_dedupe("--review-threshold", "0.5")
        self.assertEqual(res.exit_code, 0, res.output)
        path = rv.default_review_path(self.db)
        with rv.ReviewDB(path) as review:
            self.assertEqual(review.counts()["pending"], 1)
            g = review.groups()[0]
            self.assertEqual(g.n, 3)
            self.assertEqual({m.path: m.keep for m in review.members(g.id)}, {"a.png": False, "b.png": False, "c.png": True})
            self.assertEqual(review.get_meta("model_id"), "m")
            self.assertEqual(review.auto_threshold, 0.1)
        res = CliRunner().invoke(cli, ["review-status", "-d", self.db])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("pending  1", res.output)
        self.assertIn("0 files would be moved", res.output)

    def test_review_threshold_rejects_trash_dir_and_lower_value(self):
        self.assertNotEqual(self.run_dedupe("--review-threshold", "0.5", "--trash-dir", self.tmp.name).exit_code, 0)
        self.assertNotEqual(self.run_dedupe("--review-threshold", "0.05").exit_code, 0)

    def test_review_apply_requires_database(self):
        res = CliRunner().invoke(cli, ["review-apply", "-i", self.img, "-d", self.db, "--trash-dir", os.path.join(self.tmp.name, "t")])
        self.assertNotEqual(res.exit_code, 0)
        self.assertIn("No review database", res.output)
