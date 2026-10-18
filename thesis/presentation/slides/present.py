"""Start the manim-slides presenter with a fix for high-resolution renders.

manim-slides 5.7 (and its main branch) sets the presenter's image label to a minimum
size equal to the video resolution. The label shares a QStackedWidget with the video,
so a 4K render forces the whole window content to at least 3840x2160: on a 1080p
screen only the top-left quarter is visible. This launcher removes that minimum (the
video widget scales on its own, keeping the aspect ratio) and opens windowed mode at a
size that fits the screen. All `manim-slides present` options work unchanged.

Run (from thesis/presentation/slides/):
    uv run python present.py -F Intro Problem Data Generators Bands Models Close
"""

from manim_slides.present import player as _player
from manim_slides.present import present
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

_original_init = _player.Player.__init__


def _patched_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self.image_label.setMinimumSize(1, 1)
    if not self.windowState() & Qt.WindowState.WindowFullScreen:
        # Windowed: 80 % of the screen width at 16:9 instead of the render resolution.
        area = (self.screen() or QApplication.primaryScreen()).availableGeometry()
        w = int(area.width() * 0.8)
        h = min(w * 9 // 16, int(area.height() * 0.9))
        self.resize(h * 16 // 9, h)


_player.Player.__init__ = _patched_init

if __name__ == "__main__":
    present(prog_name="present.py")
