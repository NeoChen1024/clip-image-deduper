"""The review window: group list, comparison canvas, member panel, decisions and Apply."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import humanize
import PIL.Image
from PySide6.QtCore import QEvent, QSize, Qt, QThreadPool
from PySide6.QtGui import QAction, QActionGroup, QIcon, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..keeping import Policy, load_policies
from ..review import GroupRow, MemberRow, ReviewDB, apply_decisions, unapply_group
from .canvas import CompareCanvas
from .jobs import LRU, Job, diff_images, load_image, qimage, thumbnail
from .models import GROUP_ROLE, STATUS_BADGES, BadgeDelegate, GroupListModel

logger = logging.getLogger(__name__)

THUMB = (96, 96)
FILTERS = ("all", "pending", "decided", "skipped", "applied", "stale")


class ReviewWindow(QMainWindow):
    """One review database, one image directory."""

    def __init__(self, review_path: str, image_dir: str | None = None, trash_dir: str | None = None, policy: Policy | None = None):
        super().__init__()
        self.review = ReviewDB(review_path)
        self.image_dir = image_dir or self.review.get_meta("image_dir") or "."
        self.trash_dir = trash_dir
        if policy is None:
            policies = load_policies()
            policy = policies.get(self.review.get_meta("policy") or "largest") or policies["largest"]
        self.policy = policy
        self.setWindowTitle(f"Duplicate review — {os.path.basename(review_path)}")
        self.resize(1500, 950)

        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(4)
        self.images: LRU = LRU(8)  # path -> (PIL image, QImage)
        self.thumbs: LRU = LRU(300)  # path -> QPixmap
        self.pending_loads: set[str] = set()
        self.generation = 0

        self.current: GroupRow | None = None
        self.members: list[MemberRow] = []
        self.a = 0
        self.b = 1
        self.diff_key: tuple[str, str] | None = None

        self._build_toolbar()
        self._build_panes()
        self._build_statusbar()
        app = QApplication.instance()
        assert app is not None
        app.installEventFilter(self)
        self.reload_groups()

    # -- construction ------------------------------------------------------------------------------------------------

    def _action(self, toolbar: QToolBar, label: str, fn: Callable, shortcut: str | None = None, checkable: bool = False) -> QAction:
        action = QAction(label, self)
        action.triggered.connect(fn)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.setCheckable(checkable)
        toolbar.addAction(action)
        return action

    def _build_toolbar(self) -> None:
        bar = QToolBar()
        bar.setMovable(False)
        self.addToolBar(bar)
        bar.addWidget(QLabel(" Show "))
        self.filter = QComboBox()
        self.filter.addItems(FILTERS)
        self.filter.setCurrentText("all")
        self.filter.currentTextChanged.connect(lambda _: self.reload_groups())
        bar.addWidget(self.filter)
        bar.addSeparator()
        self.mode_group = QActionGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_actions: dict[str, QAction] = {}
        for label, mode in (("Side by side", "side"), ("Flip", "flip"), ("Diff", "diff")):
            action = self._action(bar, label, lambda checked=False, m=mode: self.canvas.set_mode(m), checkable=True)
            self.mode_group.addAction(action)
            self.mode_actions[mode] = action
        self.mode_actions["side"].setChecked(True)
        bar.addSeparator()
        self.zoom_group = QActionGroup(self)
        self.zoom_group.setExclusive(True)
        self.zoom_group.setExclusionPolicy(QActionGroup.ExclusionPolicy.ExclusiveOptional)  # neither is checked after a manual zoom
        # Clicking the depressed button would release it (ExclusiveOptional); the handlers re-assert the state.
        self.zoom_actions = {
            "fit": self._action(bar, "Fit", lambda: (self.canvas.fit(), self._zoom_changed("fit")), checkable=True),
            "actual": self._action(bar, "100%", lambda: (self.canvas.actual_size(), self._zoom_changed("actual")), checkable=True),
        }
        for action in self.zoom_actions.values():
            self.zoom_group.addAction(action)
        self.zoom_actions["fit"].setChecked(True)
        bar.addSeparator()
        self._action(bar, "Previous (Z)", lambda: self.navigate(-1), "Alt+Left")
        self._action(bar, "Next (X)", lambda: self.navigate(1), "Alt+Right")
        self._action(bar, "Policy pick (P)", self.reset_policy)
        self._action(bar, "Undo (U)", self.undo)
        bar.addSeparator()
        self._action(bar, "Apply…", self.apply_dialog, "Ctrl+Return")
        self._action(bar, "Undo apply", self.undo_apply)
        bar.addSeparator()
        self._action(bar, "Help", self.show_help, "F1")

    def _build_panes(self) -> None:
        self.groups_model = GroupListModel()
        self.group_list = QListView()
        self.group_list.setModel(self.groups_model)
        self.group_list.setItemDelegate(BadgeDelegate(self.group_list))
        self.group_list.setUniformItemSizes(True)
        self.group_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.group_list.setMinimumWidth(170)
        self.group_list.setMaximumWidth(320)
        self.group_list.selectionModel().currentChanged.connect(self._group_selected)

        self.canvas = CompareCanvas()
        self.canvas.navigate.connect(self.navigate)
        self.canvas.mode_changed.connect(self._mode_changed)
        self.canvas.zoom_changed.connect(self._zoom_changed)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.group_label = QLabel("No group")
        self.group_label.setWordWrap(True)
        layout.addWidget(self.group_label)
        layout.addWidget(QLabel("Members · number key shows as A, Shift+number as B · checkbox = keep"))
        self.member_list = QListWidget()
        self.member_list.setIconSize(QSize(*THUMB))
        self.member_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.member_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.member_list.itemChanged.connect(self._member_checked)
        self.member_list.itemClicked.connect(self._member_clicked)
        layout.addWidget(self.member_list, 1)
        self.edges_label = QLabel()
        self.edges_label.setWordWrap(True)
        self.edges_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(QLabel("Pair distances"))
        layout.addWidget(self.edges_label)
        layout.addWidget(QLabel("Note (F)"))
        self.note = QLineEdit()
        self.note.editingFinished.connect(self._note_edited)
        layout.addWidget(self.note)
        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        split = QSplitter()
        split.addWidget(self.group_list)
        split.addWidget(self.canvas)
        split.addWidget(panel)
        split.setSizes([220, 880, 400])
        self.setCentralWidget(split)

    def _build_statusbar(self) -> None:
        self.position_label = QLabel("Group 0 / 0")
        self.position_label.setContentsMargins(6, 0, 12, 0)
        self.counts_label = QLabel()
        self.message_label = QLabel()
        self.statusBar().addPermanentWidget(self.position_label)
        self.statusBar().addPermanentWidget(self.counts_label)
        self.statusBar().addPermanentWidget(self.message_label, 1)
        self.message("Ready · nothing is deleted, Apply moves files to the trash directory")

    # -- groups --------------------------------------------------------------------------------------------------------

    def reload_groups(self, select_id: int | None = None) -> None:
        """Re-read the group list for the current filter; keep (or set) the selection."""
        if select_id is None and self.current is not None:
            select_id = self.current.id
        status = self.filter.currentText()
        groups = self.review.groups(None if status == "all" else status)
        self.groups_model.set_groups(groups)
        self._update_counts()
        row = self.groups_model.row_of(select_id) if select_id is not None else -1
        if row < 0 and groups:
            row = 0
        if row >= 0:
            self.group_list.setCurrentIndex(self.groups_model.index(row))
        else:
            self.current = None
            self.members = []
            self._show_members()
            self.canvas.clear()
            self.group_label.setText("No group")
            self.position_label.setText("Group 0 / 0")

    def _update_counts(self) -> None:
        c = self.review.counts()
        moves = len(self.review.plan_apply())
        self.counts_label.setText(f"{c['pending']} pending · {c['decided']} decided · {c['skipped']} skipped · {c['applied']} applied · {moves} files to trash")

    def navigate(self, delta: int, pending_only: bool = False) -> None:
        n = self.groups_model.rowCount()
        if n == 0:
            return
        row = self.group_list.currentIndex().row()
        if pending_only:
            candidates = range(row + 1, n) if delta > 0 else range(row - 1, -1, -1)
            for r in candidates:
                if self.groups_model.groups[r].status == "pending":
                    self.group_list.setCurrentIndex(self.groups_model.index(r))
                    return
            self.message("No more pending groups in this direction")
            return
        row = max(0, min(n - 1, row + delta))
        self.group_list.setCurrentIndex(self.groups_model.index(row))

    def _group_selected(self, index, previous) -> None:
        if not index.isValid():
            return
        group = index.data(GROUP_ROLE)
        self.position_label.setText(f"Group {index.row() + 1:,} / {self.groups_model.rowCount():,}")
        self.show_group(group)

    def show_group(self, group: GroupRow) -> None:
        self.current = group
        self.members = self.review.members(group.id)
        keeps = [i for i, m in enumerate(self.members) if m.keep]
        self.a = keeps[0] if keeps else 0
        edges = self.review.edges(group.id)
        self.b = self._nearest(self.a, edges)
        self.diff_key = None
        self.note.blockSignals(True)
        self.note.setText(group.note)
        self.note.blockSignals(False)
        self._show_members()
        names = {m.path: str(i + 1) for i, m in enumerate(self.members)}
        self.edges_label.setText("  ".join(f"{names.get(a, '?')}–{names.get(b, '?')}: {d:.3f}" for a, b, d in edges[:30]) + ("  …" if len(edges) > 30 else ""))
        self._refresh_group_label()
        self._request_images()

    def _nearest(self, i: int, edges: list[tuple[str, str, float]]) -> int:
        """The member closest to member ``i`` by the stored edges (edges are sorted by distance), else another one."""
        index = {m.path: k for k, m in enumerate(self.members)}
        me = self.members[i].path
        for a, b, _ in edges:
            if a == me and b in index:
                return index[b]
            if b == me and a in index:
                return index[a]
        return next((k for k in range(len(self.members)) if k != i), i)

    def _refresh_group_label(self) -> None:
        g = self.current
        if g is None:
            return
        label = STATUS_BADGES.get(g.status, STATUS_BADGES["pending"])[1]
        self.group_label.setText(f"Group {g.id}: {g.n} images · distances {g.min_distance:.3f} – {g.max_distance:.3f}")
        kept = sum(1 for m in self.members if m.keep)
        self.status_label.setText(f"Status: {label} (C cycles) · keeping {kept} of {len(self.members)}")

    # -- members -------------------------------------------------------------------------------------------------------

    def _member_text(self, i: int) -> str:
        m = self.members[i]
        tag = "A" if i == self.a else "B" if i == self.b else " "
        dist = "" if m.distance_to_winner is None else f" · d={m.distance_to_winner:.3f}"
        return f"{i + 1} [{tag}] {m.path}\n{m.width}×{m.height} · {m.format} · {humanize.naturalsize(m.size, binary=True)}{dist}"

    def _refresh_members(self) -> None:
        """Update labels and check boxes in place (never rebuild the list from inside one of its own signals)."""
        self.member_list.blockSignals(True)
        for i in range(min(self.member_list.count(), len(self.members))):
            item = self.member_list.item(i)
            item.setText(self._member_text(i))
            item.setCheckState(Qt.CheckState.Checked if self.members[i].keep else Qt.CheckState.Unchecked)
        self.member_list.blockSignals(False)

    def _show_members(self) -> None:
        self.member_list.blockSignals(True)
        self.member_list.clear()
        for i, m in enumerate(self.members):
            item = QListWidgetItem(self._member_text(i))
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if m.keep else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, i)
            pix = self.thumbs.get(m.path)
            if pix is not None:
                item.setIcon(QIcon(pix))
            else:
                self._request_thumbnail(m.path)
            self.member_list.addItem(item)
        self.member_list.blockSignals(False)
        if 0 <= self.a < self.member_list.count():
            self.member_list.scrollToItem(self.member_list.item(self.a), QAbstractItemView.ScrollHint.PositionAtCenter)

    def _member_checked(self, item: QListWidgetItem) -> None:
        if self.current is None:
            return
        i = item.data(Qt.ItemDataRole.UserRole)
        keep = item.checkState() == Qt.CheckState.Checked
        self._decide(keeps={self.members[i].path: keep})

    def _member_clicked(self, item: QListWidgetItem | None) -> None:
        if item is None:
            return
        i = item.data(Qt.ItemDataRole.UserRole)
        if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.set_b(i)
        else:
            self.set_a(i)

    def set_a(self, i: int) -> None:
        if 0 <= i < len(self.members) and i != self.a:
            if i == self.b:
                self.b = self.a
            self.a = i
            self._refresh_members()
            self._request_images()

    def set_b(self, i: int) -> None:
        if 0 <= i < len(self.members) and i != self.b:
            if i == self.a:
                self.a = self.b
            self.b = i
            self._refresh_members()
            self._request_images()

    # -- images --------------------------------------------------------------------------------------------------------

    def _request_thumbnail(self, path: str) -> None:
        if path in self.pending_loads:
            return
        self.pending_loads.add(path)
        full = os.path.join(self.image_dir, path)

        def read(signals):
            return path, thumbnail(full, THUMB)

        self.pool.start(Job(read).connect(self._thumbnail_ready, lambda error, p=path: self._thumbnail_failed(p, error)))

    def _thumbnail_ready(self, result) -> None:
        path, image = result
        self.pending_loads.discard(path)
        self.thumbs.put(path, QPixmap.fromImage(image))
        for row in range(self.member_list.count()):
            item = self.member_list.item(row)
            if self.members and self.members[item.data(Qt.ItemDataRole.UserRole)].path == path:
                pix = self.thumbs.get(path)
                if pix is not None:
                    item.setIcon(QIcon(pix))

    def _thumbnail_failed(self, path: str, error: str) -> None:
        self.pending_loads.discard(path)
        self.thumbs.put(path, QPixmap())
        self.message(f"Cannot read {path}: {error}")

    def _request_images(self) -> None:
        """Load A and B (cached or off-thread) and show them when both are in."""
        self.generation += 1
        generation = self.generation
        if not self.members:
            self.canvas.clear()
            return
        paths = [self.members[self.a].path, self.members[self.b].path if self.b != self.a else None]
        for path in paths:
            if path is None or path in self.images:
                continue
            full = os.path.join(self.image_dir, path)

            def read(signals, p=path, f=full):
                im = load_image(f)
                return p, im, qimage(im)

            self.pool.start(
                Job(read).connect(lambda result, g=generation: self._image_loaded(result, g), lambda error, p=path, g=generation: self._image_failed(p, error, g))
            )
        self._show_images(generation)

    def _image_loaded(self, result, generation: int) -> None:
        path, im, image = result
        self.images.put(path, (im, image))
        self._show_images(generation)

    def _image_failed(self, path: str, error: str, generation: int) -> None:
        self.images.put(path, (None, QImage()))
        self.message(f"Cannot read {path}: {error}")
        self._show_images(generation)

    def _show_images(self, generation: int) -> None:
        if generation != self.generation or not self.members:
            return
        pa = self.members[self.a].path
        pb = self.members[self.b].path if self.b != self.a else None
        ca = self.images.get(pa)
        cb = self.images.get(pb) if pb else (None, None)
        if ca is None or (pb and cb is None):
            return  # still loading
        self.canvas.set_images(ca[1], cb[1] if cb else None)
        if self.canvas.mode == "diff":
            self._request_diff()

    def _request_diff(self) -> None:
        if not self.members or self.b == self.a:
            self.canvas.set_diff(None)
            return
        pa, pb = self.members[self.a].path, self.members[self.b].path
        if self.diff_key == (pa, pb):
            return
        ca, cb = self.images.get(pa), self.images.get(pb)
        if ca is None or cb is None or ca[0] is None or cb[0] is None:
            return
        self.diff_key = (pa, pb)
        generation = self.generation
        a_im, b_im = ca[0], cb[0]

        def compute(signals):
            return diff_images(a_im, b_im)

        self.pool.start(Job(compute).connect(lambda images, g=generation, k=(pa, pb): self._diff_ready(images, g, k), lambda error: self.message(f"Diff failed: {error}")))

    def _diff_ready(self, images, generation: int, key) -> None:
        if generation == self.generation and self.diff_key == key:
            self.canvas.set_diff(*images)

    def _zoom_changed(self, state: str) -> None:
        for name, action in self.zoom_actions.items():
            action.setChecked(name == state)

    def _mode_changed(self, mode: str) -> None:
        self.mode_actions[mode].setChecked(True)
        if mode == "diff":
            self._request_diff()

    def wait_idle(self) -> None:
        """Block until background jobs have finished and their results were delivered (for tests)."""
        self.pool.waitForDone()
        QApplication.processEvents()
        QApplication.processEvents()

    # -- decisions -----------------------------------------------------------------------------------------------------

    def _decide(self, **kwargs) -> None:
        if self.current is None:
            return
        try:
            self.review.decide(self.current.id, **kwargs)
        except ValueError as e:
            self.message(str(e))
            return
        self._after_change()

    def _after_change(self) -> None:
        assert self.current is not None
        group = self.review.group(self.current.id)
        self.current = group
        self.members = self.review.members(group.id)
        self.groups_model.update_group(group)
        self._refresh_members()
        self._refresh_group_label()
        self._update_counts()
        self.note.blockSignals(True)
        self.note.setText(group.note)
        self.note.blockSignals(False)

    def toggle_keep_a(self) -> None:
        if self.members:
            m = self.members[self.a]
            self._decide(keeps={m.path: not m.keep})

    def keep_only_a(self) -> None:
        if self.members:
            self._decide(keeps={m.path: i == self.a for i, m in enumerate(self.members)})

    def reset_policy(self) -> None:
        if self.current is None:
            return
        try:
            self.review.reset_to_policy(self.current.id, self.policy)
        except ValueError as e:
            self.message(str(e))
            return
        self._after_change()

    def decide_and_next(self) -> None:
        if self.current is None:
            return
        if not any(m.keep for m in self.members):
            self.message("Nothing is kept in this group; tick at least one member before deciding")
            return
        self._decide(status="decided")
        self.navigate(1, pending_only=True)

    def cycle_status(self) -> None:
        if self.current is None or self.current.status in ("applied", "stale"):
            return
        cycle = ("pending", "decided", "skipped")
        self._decide(status=cycle[(cycle.index(self.current.status) + 1) % 3])

    def skip_and_next(self) -> None:
        if self.current is None:
            return
        self._decide(status="skipped")
        self.navigate(1, pending_only=True)

    def undo(self) -> None:
        if self.current is None:
            return
        try:
            if not self.review.undo(self.current.id):
                self.message("Nothing to undo for this group")
                return
        except ValueError as e:
            self.message(str(e))
            return
        self._after_change()

    def _note_edited(self) -> None:
        if self.current is not None and self.note.text() != self.current.note:
            self._decide(note=self.note.text())

    # -- apply ---------------------------------------------------------------------------------------------------------

    def plan_text(self) -> tuple[str, int]:
        moves = self.review.plan_apply()
        lines = [f"{m.path}  →  trash   (keeping {', '.join(m.keep)})" for m in moves]
        return "\n".join(lines), len(moves)

    def _ensure_trash_dir(self) -> bool:
        if self.trash_dir:
            return True
        chosen = QFileDialog.getExistingDirectory(self, "Trash directory (losers are moved here)")
        if not chosen:
            return False
        self.trash_dir = chosen
        return True

    def apply_dialog(self) -> None:
        text, n = self.plan_text()
        if n == 0:
            QMessageBox.information(self, "Nothing to apply", "No decided group has files to move.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Apply decisions")
        dialog.resize(800, 600)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"{n} files will be moved to the trash directory{': ' + self.trash_dir if self.trash_dir else ''}. Nothing is deleted."))
        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setPlainText(text)
        layout.addWidget(view)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(f"Move {n} files")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if not self._ensure_trash_dir():
            return
        self.apply_moves()

    def apply_moves(self, *, dry_run: bool = False) -> tuple[int, list[int]]:
        assert self.trash_dir
        decided = self.review.groups("decided")
        progress = QProgressDialog("Moving files…", "", 0, len(decided), self)
        progress.setCancelButton(None)
        progress.setMinimumDuration(500)
        done = 0

        def tick(k: int) -> None:
            nonlocal done
            done += k
            progress.setValue(done)
            QApplication.processEvents()

        moved, refused = apply_decisions(self.review, self.image_dir, self.trash_dir, dry_run=dry_run, progress=tick)
        progress.close()
        self.images.clear()
        self.reload_groups()
        self.message(f"{moved} files moved to {self.trash_dir}" + (f"; {len(refused)} groups refused because files changed on disk" if refused else ""))
        return moved, refused

    def undo_apply(self) -> None:
        if self.current is None or self.current.status != "applied":
            self.message("Select an applied group to undo its apply")
            return
        restored = unapply_group(self.review, self.image_dir, self.current.id)
        self.images.clear()
        self._after_change()
        self.message(f"{restored} files restored")

    # -- keys ----------------------------------------------------------------------------------------------------------

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt API
        if event.type() not in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) or QApplication.activeWindow() is not self:
            return super().eventFilter(obj, event)
        widget = QApplication.focusWidget()
        while widget is not None:
            if isinstance(widget, (QPlainTextEdit, QTextEdit, QLineEdit, QAbstractSpinBox)):
                if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
                    self.canvas.left.setFocus()
                    return True
                return super().eventFilter(obj, event)
            widget = widget.parentWidget()
        key = event.key()
        mods = event.modifiers()
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        plain = mods in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ShiftModifier, Qt.KeyboardModifier.KeypadModifier)
        if key == Qt.Key.Key_Space and plain:
            if not event.isAutoRepeat():
                self.canvas.set_flipped(event.type() == QEvent.Type.KeyPress)
            return True
        if event.type() != QEvent.Type.KeyPress or not plain:
            return super().eventFilter(obj, event)
        if key == Qt.Key.Key_Z:
            self.navigate(-1, pending_only=shift)
        elif key == Qt.Key.Key_X:
            self.navigate(1, pending_only=shift)
        elif Qt.Key.Key_1 <= key <= Qt.Key.Key_9 or (shift and key in _SHIFTED_DIGITS):
            n = _SHIFTED_DIGITS.get(key, key - Qt.Key.Key_1 + 1)
            (self.set_b if shift else self.set_a)(n - 1)
        elif key == Qt.Key.Key_D:
            self.canvas.cycle_mode()
        elif key == Qt.Key.Key_K:
            self.keep_only_a() if shift else self.toggle_keep_a()
        elif key == Qt.Key.Key_P:
            self.reset_policy()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.decide_and_next()
        elif key == Qt.Key.Key_C:
            if not event.isAutoRepeat():
                self.cycle_status()
        elif key == Qt.Key.Key_S:
            self.skip_and_next()
        elif key == Qt.Key.Key_U:
            self.undo()
        elif key == Qt.Key.Key_F:
            self.note.setFocus()
            self.note.selectAll()
        elif key == Qt.Key.Key_Home:
            self.canvas.fit()
        elif key == Qt.Key.Key_0:
            self.canvas.actual_size()
        else:
            return super().eventFilter(obj, event)
        return True

    # -- misc ----------------------------------------------------------------------------------------------------------

    def message(self, text: str) -> None:
        self.message_label.setText(text)
        logger.info(text)

    def show_help(self) -> None:
        if not hasattr(self, "help_dialog"):
            self.help_dialog = QDialog(self)
            self.help_dialog.setWindowTitle("Help — Keyboard Shortcuts")
            self.help_dialog.resize(640, 640)
            layout = QVBoxLayout(self.help_dialog)
            browser = QTextBrowser()
            browser.setHtml(HELP_HTML)
            layout.addWidget(browser)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            buttons.rejected.connect(self.help_dialog.reject)
            layout.addWidget(buttons)
        self.help_dialog.show()
        self.help_dialog.raise_()
        self.help_dialog.activateWindow()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.pool.waitForDone()
        self.review.close()
        super().closeEvent(event)


_SHIFTED_DIGITS = {
    Qt.Key.Key_Exclam: 1, Qt.Key.Key_At: 2, Qt.Key.Key_NumberSign: 3, Qt.Key.Key_Dollar: 4, Qt.Key.Key_Percent: 5,
    Qt.Key.Key_AsciiCircum: 6, Qt.Key.Key_Ampersand: 7, Qt.Key.Key_Asterisk: 8, Qt.Key.Key_ParenLeft: 9,
}

HELP_HTML = """
<h2>Keyboard Shortcuts</h2>
<p>Single letters are ignored while the note field has focus; Esc returns to the canvas.</p>
<h3>Groups</h3>
<table cellspacing="6">
<tr><td><b>Z / X</b>, wheel, Alt+Left / Right</td><td>Previous / next group</td></tr>
<tr><td><b>Shift+Z / Shift+X</b></td><td>Previous / next <i>pending</i> group</td></tr>
<tr><td><b>Enter</b></td><td>Mark decided and go to the next pending group</td></tr>
<tr><td><b>S</b></td><td>Skip and go to the next pending group</td></tr>
<tr><td><b>C</b></td><td>Cycle pending → decided → skipped</td></tr>
<tr><td><b>U</b></td><td>Undo the last change of this group</td></tr>
<tr><td><b>F</b></td><td>Edit the note</td></tr>
</table>
<h3>Members</h3>
<table cellspacing="6">
<tr><td><b>1-9</b></td><td>Show member n as A (click does the same)</td></tr>
<tr><td><b>Shift+1-9</b></td><td>Show member n as B (Shift+click)</td></tr>
<tr><td><b>K</b></td><td>Toggle keep on A</td></tr>
<tr><td><b>Shift+K</b></td><td>Keep only A</td></tr>
<tr><td><b>P</b></td><td>Reset keeps to the keeping policy's choice</td></tr>
</table>
<h3>Canvas</h3>
<table cellspacing="6">
<tr><td><b>D</b></td><td>Cycle side by side → flip → diff</td></tr>
<tr><td><b>Space</b> (hold)</td><td>Flip mode: show B instead of A. Diff mode: hide the dimmed original, differences on black</td></tr>
<tr><td><b>Ctrl+wheel</b></td><td>Zoom both views</td></tr>
<tr><td><b>Space+drag</b>, middle drag</td><td>Pan</td></tr>
<tr><td><b>Home / 0</b></td><td>Fit / 100%</td></tr>
</table>
<h3>Applying</h3>
<table cellspacing="6">
<tr><td><b>Ctrl+Enter</b></td><td>Apply: list every move, then move the losers of decided groups to the trash directory</td></tr>
<tr><td>Undo apply</td><td>Move the selected applied group's files back</td></tr>
</table>
<p>Nothing is ever deleted. Groups whose files changed on disk since matching are refused by Apply; re-run
<code>update-db</code> and <code>dedupe --review-threshold</code> for those. Decisions are written to the review
database immediately. Re-matching from the CLI keeps decisions of groups whose membership did not change.</p>
"""
