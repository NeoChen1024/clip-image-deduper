"""Offscreen checks of the review GUI. Skipped when PySide6 is not installed."""

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import PIL.Image

try:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from clip_image_deduper.gui.canvas import CompareCanvas
    from clip_image_deduper.gui.window import ReviewWindow
except ImportError:  # pragma: no cover
    QApplication = None  # type: ignore[assignment]

from clip_image_deduper.db_store import ImageRecord
from clip_image_deduper.keeping import builtin_policies
from clip_image_deduper.review import ReviewDB


@unittest.skipIf(QApplication is None, "PySide6 not installed")
class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.img = os.path.join(self.tmp.name, "img")
        os.makedirs(os.path.join(self.img, "sub"))
        recs = {}
        for path, colour, size in [("a.png", (255, 0, 0), (40, 30)), ("sub/b.png", (250, 0, 0), (20, 15)), ("c.png", (0, 255, 0), (40, 40)), ("d.png", (0, 250, 0), (40, 40))]:
            full = os.path.join(self.img, path)
            PIL.Image.new("RGB", size, colour).save(full)
            st = os.stat(full)
            recs[path] = ImageRecord(path, st.st_mtime, st.st_size, size[0], size[1], "PNG")
        self.review_path = os.path.join(self.tmp.name, "x.review.sqlite")
        with ReviewDB(self.review_path) as db:
            db.begin_session(image_dir=self.img, model_id="m", review_threshold=1.0, auto_threshold=0.1, policy="largest")
            db.upsert_groups(
                [([recs["a.png"], recs["sub/b.png"]], [("a.png", "sub/b.png", 0.3)]), ([recs["c.png"], recs["d.png"]], [("c.png", "d.png", 0.05)])],
                builtin_policies()["largest"],
                0.1,
            )
        self.trash = os.path.join(self.tmp.name, "trash")
        self.win = ReviewWindow(self.review_path, trash_dir=self.trash)
        self.win.show()
        self.app.setActiveWindow(self.win)
        self.win.canvas.left.setFocus()
        self.win.wait_idle()

    def tearDown(self):
        self.win.close()
        self.tmp.cleanup()

    def key(self, key, modifier=Qt.KeyboardModifier.NoModifier):
        QTest.keyClick(self.win.canvas.left, key, modifier)
        self.win.wait_idle()

    def test_groups_listed_surest_first_and_images_loaded(self):
        self.assertEqual([g.min_distance for g in self.win.groups_model.groups], [0.05, 0.3])
        self.assertEqual(self.win.current.status, "decided")  # pre-decided within 0.1
        self.assertEqual(self.win.position_label.text(), "Group 1 / 2")
        self.assertEqual(self.win.canvas.left.image_size, (40, 40))
        self.assertEqual(self.win.member_list.count(), 2)

    def test_navigation_keys(self):
        self.key(Qt.Key.Key_X)
        self.assertEqual(self.win.current.min_distance, 0.3)
        self.assertEqual(self.win.position_label.text(), "Group 2 / 2")
        self.key(Qt.Key.Key_Z)
        self.assertEqual(self.win.current.min_distance, 0.05)
        self.key(Qt.Key.Key_X, Qt.KeyboardModifier.ShiftModifier)  # next pending
        self.assertEqual(self.win.current.status, "pending")

    def test_a_b_selection_and_keep_toggle_persist(self):
        self.key(Qt.Key.Key_X)
        members = self.win.members
        self.assertEqual([m.path for m in members], ["a.png", "sub/b.png"])
        self.assertEqual((self.win.a, self.win.b), (0, 1))  # a.png is largest, so A
        self.assertEqual(self.win.canvas.left.image_size, (40, 30))
        self.assertEqual(self.win.canvas.right.image_size, (20, 15))
        self.key(Qt.Key.Key_2)
        self.assertEqual((self.win.a, self.win.b), (1, 0))
        self.key(Qt.Key.Key_K)
        with ReviewDB(self.review_path) as db:
            self.assertEqual({m.path: m.keep for m in db.members(self.win.current.id)}, {"a.png": True, "sub/b.png": True})
        self.key(Qt.Key.Key_K, Qt.KeyboardModifier.ShiftModifier)
        with ReviewDB(self.review_path) as db:
            self.assertEqual({m.path: m.keep for m in db.members(self.win.current.id)}, {"a.png": False, "sub/b.png": True})
        self.key(Qt.Key.Key_P)
        with ReviewDB(self.review_path) as db:
            self.assertEqual({m.path: m.keep for m in db.members(self.win.current.id)}, {"a.png": True, "sub/b.png": False})
        self.key(Qt.Key.Key_U)
        with ReviewDB(self.review_path) as db:
            self.assertEqual({m.path: m.keep for m in db.members(self.win.current.id)}, {"a.png": False, "sub/b.png": True})

    def test_enter_decides_and_skip_cycle(self):
        self.key(Qt.Key.Key_X)
        gid = self.win.current.id
        self.key(Qt.Key.Key_Return)
        with ReviewDB(self.review_path) as db:
            self.assertEqual(db.group(gid).status, "decided")
        self.key(Qt.Key.Key_C)  # decided -> skipped
        with ReviewDB(self.review_path) as db:
            self.assertEqual(db.group(gid).status, "skipped")
        self.assertIn("Skipped", self.win.status_label.text())

    def test_modes_flip_and_diff(self):
        self.key(Qt.Key.Key_X)
        self.key(Qt.Key.Key_D)
        self.assertEqual(self.win.canvas.mode, "flip")
        self.assertEqual(self.win.canvas.left.image_size, (40, 30))
        QTest.keyPress(self.win.canvas.left, Qt.Key.Key_Space)
        self.assertEqual(self.win.canvas.left.image_size, (20, 15))
        QTest.keyRelease(self.win.canvas.left, Qt.Key.Key_Space)
        self.assertEqual(self.win.canvas.left.image_size, (40, 30))
        self.key(Qt.Key.Key_D)
        self.assertEqual(self.win.canvas.mode, "diff")
        self.win.wait_idle()
        self.assertEqual(self.win.canvas.left.image_size, (40, 30))
        self.assertFalse(self.win.canvas.right.isVisible())
        overlay = self.win.canvas.left.pix.pixmap().toImage()
        self.assertGreater(overlay.pixelColor(0, 0).red(), 100)  # the dimmed red original shows through
        QTest.keyPress(self.win.canvas.left, Qt.Key.Key_Space)
        plain = self.win.canvas.left.pix.pixmap().toImage()
        self.assertLess(plain.pixelColor(0, 0).red(), 60)  # on black only the amplified difference (5 * 4) remains
        QTest.keyRelease(self.win.canvas.left, Qt.Key.Key_Space)
        self.assertGreater(self.win.canvas.left.pix.pixmap().toImage().pixelColor(0, 0).red(), 100)
        self.key(Qt.Key.Key_D)
        self.assertEqual(self.win.canvas.mode, "side")

    def test_letters_ignored_in_note_field_and_note_saved(self):
        self.key(Qt.Key.Key_X)
        self.key(Qt.Key.Key_F)
        self.assertTrue(self.win.note.hasFocus())
        QTest.keyClicks(self.win.note, "xz keep both")
        QTest.keyClick(self.win.note, Qt.Key.Key_Return)
        self.win.wait_idle()
        self.assertEqual(self.win.position_label.text(), "Group 2 / 2")  # x/z did not navigate
        with ReviewDB(self.review_path) as db:
            self.assertEqual(db.group(self.win.current.id).note, "xz keep both")
        QTest.keyClick(self.win.note, Qt.Key.Key_Escape)
        self.assertFalse(self.win.note.hasFocus())

    def test_apply_moves_losers(self):
        text, n = self.win.plan_text()
        self.assertEqual(n, 1)
        self.assertIn("c.png", text)  # d.png is the same size; the stable sort keeps the first, c... no: largest wins, tie keeps input order
        moved, refused = self.win.apply_moves()
        self.assertEqual((moved, refused), (1, []))
        self.assertTrue(os.path.exists(os.path.join(self.trash, "c.png")) or os.path.exists(os.path.join(self.trash, "d.png")))
        self.assertEqual(self.win.current.status, "applied")
        self.win.undo_apply()
        self.assertEqual(self.win.current.status, "decided")
        self.assertTrue(os.path.exists(os.path.join(self.img, "c.png")) and os.path.exists(os.path.join(self.img, "d.png")))

    def test_b_defaults_to_nearest_of_a(self):
        with ReviewDB(self.review_path) as db:
            recs = {m.path: m.record for g in db.groups() for m in db.members(g.id)}
            db.upsert_groups(
                [([recs["a.png"], recs["sub/b.png"], recs["c.png"]], [("a.png", "sub/b.png", 0.9), ("sub/b.png", "c.png", 0.2)])],
                builtin_policies()["largest"], 0.1,
            )
        self.win.reload_groups()
        self.win.wait_idle()
        self.assertEqual(self.win.current.n, 3)
        self.assertEqual([m.path for m in self.win.members], ["a.png", "c.png", "sub/b.png"])
        self.assertEqual(self.win.members[self.win.a].path, "c.png")  # largest pixel count... size: a.png 40x30 png vs c 40x40
        self.assertEqual(self.win.members[self.win.b].path, "sub/b.png")  # nearest to c.png (0.2), not member 1

    def test_checkbox_click_does_not_rebuild_list(self):
        self.key(Qt.Key.Key_X)
        item = self.win.member_list.item(1)
        item.setCheckState(Qt.CheckState.Checked)  # emits itemChanged like a click on the box
        self.win._member_clicked(item)  # the click that follows must still find the same item
        self.win._member_clicked(None)  # and a stray click on no item must not raise
        self.assertIs(self.win.member_list.item(1), item)
        with ReviewDB(self.review_path) as db:
            self.assertEqual({m.path: m.keep for m in db.members(self.win.current.id)}, {"a.png": True, "sub/b.png": True})

    def test_filter(self):
        self.win.filter.setCurrentText("pending")
        self.win.wait_idle()
        self.assertEqual(self.win.groups_model.rowCount(), 1)
        self.assertEqual(self.win.current.status, "pending")


@unittest.skipIf(QApplication is None, "PySide6 not installed")
class CanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_zoom_applies_to_both_views(self):
        from clip_image_deduper.gui.jobs import qimage

        canvas = CompareCanvas()
        canvas.resize(400, 200)
        canvas.show()
        canvas.set_images(qimage(PIL.Image.new("RGB", (100, 50))), qimage(PIL.Image.new("RGB", (50, 50))))
        canvas.zoom(2.0)
        self.assertAlmostEqual(canvas.left.transform().m11(), canvas.right.transform().m11())
        canvas.actual_size()
        self.assertEqual(canvas.scale_factor(), 1.0)
        canvas.close()
