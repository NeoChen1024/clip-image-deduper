"""The comparison canvas: two image views sharing one zoom/pan, with flip and difference modes."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QHBoxLayout, QWidget

MODES = ("side", "flip", "diff")


class ImageView(QGraphicsView):
    """One pixmap in a scene; zoom/pan requests are forwarded to the canvas so both views move together."""

    wheel = Signal(object)  # the QWheelEvent
    space_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#252525"))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(self.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(self.ViewportAnchor.AnchorViewCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.pix = self.scene().addPixmap(QPixmap())
        self.image_size = (0, 0)

    def set_image(self, image: QImage | None) -> None:
        if image is None or image.isNull():
            self.pix.setPixmap(QPixmap())
            self.image_size = (0, 0)
            self.scene().setSceneRect(QRectF())
            return
        self.pix.setPixmap(QPixmap.fromImage(image))
        self.image_size = (image.width(), image.height())
        self.scene().setSceneRect(0, 0, image.width(), image.height())

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.wheel.emit(event)
        event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.space_changed.emit(True)
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.space_changed.emit(False)
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            # Re-dispatch as a left press so the hand drag starts.
            fake = type(event)(event.type(), event.position(), event.globalPosition(), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, event.modifiers())
            super().mousePressEvent(fake)
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton:
            fake = type(event)(event.type(), event.position(), event.globalPosition(), Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, event.modifiers())
            super().mouseReleaseEvent(fake)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            return
        super().mouseReleaseEvent(event)


class CompareCanvas(QWidget):
    """Two linked :class:`ImageView`s.

    ``side``: A left, B right. ``flip``: one view showing A, or B while ``flipped`` is held. ``diff``: one view
    showing the amplified absolute difference (supplied by the window, computed off-thread).
    """

    navigate = Signal(int)  # +1 / -1 from the wheel
    mode_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.left = ImageView()
        self.right = ImageView()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.left)
        layout.addWidget(self.right)
        self.mode = "side"
        self.flipped = False
        self.a: QImage | None = None
        self.b: QImage | None = None
        self.diff: QImage | None = None
        self._syncing = False
        for view in (self.left, self.right):
            view.wheel.connect(self._wheel)
            view.space_changed.connect(self._space)
        for a, b in ((self.left, self.right), (self.right, self.left)):
            a.horizontalScrollBar().valueChanged.connect(lambda v, other=b: self._sync_scroll(other, "h", v))
            a.verticalScrollBar().valueChanged.connect(lambda v, other=b: self._sync_scroll(other, "v", v))

    # -- content ---------------------------------------------------------------------------------------------------

    def set_images(self, a: QImage | None, b: QImage | None, *, fit: bool = True) -> None:
        self.a, self.b, self.diff = a, b, None
        self._show()
        if fit:
            self.fit()

    def set_diff(self, diff: QImage | None) -> None:
        self.diff = diff
        if self.mode == "diff":
            self._show()

    def clear(self) -> None:
        self.set_images(None, None)

    def _show(self) -> None:
        if self.mode == "side":
            self.right.show()
            self.left.set_image(self.a)
            self.right.set_image(self.b)
        elif self.mode == "flip":
            self.right.hide()
            self.left.set_image(self.b if self.flipped else self.a)
        else:
            self.right.hide()
            self.left.set_image(self.diff)

    # -- modes -----------------------------------------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(mode)
        if mode == self.mode:
            return
        transform = self.left.transform()
        self.mode = mode
        self._show()
        self.left.setTransform(transform)
        self.right.setTransform(transform)
        self.mode_changed.emit(mode)

    def cycle_mode(self) -> str:
        self.set_mode(MODES[(MODES.index(self.mode) + 1) % len(MODES)])
        return self.mode

    def set_flipped(self, flipped: bool) -> None:
        if flipped == self.flipped:
            return
        self.flipped = flipped
        if self.mode == "flip":
            transform = self.left.transform()
            h, v = self.left.horizontalScrollBar().value(), self.left.verticalScrollBar().value()
            self.left.set_image(self.b if flipped else self.a)
            self.left.setTransform(transform)
            self.left.horizontalScrollBar().setValue(h)
            self.left.verticalScrollBar().setValue(v)

    # -- zoom / pan ------------------------------------------------------------------------------------------------

    def fit(self) -> None:
        for view in (self.left, self.right):
            if view.image_size != (0, 0):
                view.fitInView(view.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        # Both views share the smaller scale so the pictures line up.
        scale = min(v.transform().m11() for v in (self.left, self.right) if v.image_size != (0, 0)) if self.a or self.b else 1.0
        for view in (self.left, self.right):
            view.resetTransform()
            view.scale(scale, scale)

    def actual_size(self) -> None:
        for view in (self.left, self.right):
            view.resetTransform()

    def zoom(self, factor: float) -> None:
        for view in (self.left, self.right):
            view.scale(factor, factor)

    def scale_factor(self) -> float:
        return self.left.transform().m11()

    def _wheel(self, event) -> None:
        delta = event.angleDelta().y()
        if not delta:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom(1.2 ** (delta / 120))
        else:
            self.navigate.emit(-1 if delta > 0 else 1)

    def _space(self, down: bool) -> None:
        mode = QGraphicsView.DragMode.ScrollHandDrag if down else QGraphicsView.DragMode.NoDrag
        for view in (self.left, self.right):
            view.setDragMode(mode)

    def _sync_scroll(self, other: ImageView, axis: str, value: int) -> None:
        if self._syncing or self.mode != "side":
            return
        self._syncing = True
        try:
            bar = other.horizontalScrollBar() if axis == "h" else other.verticalScrollBar()
            bar.setValue(value)
        finally:
            self._syncing = False
