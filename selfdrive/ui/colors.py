#!/usr/bin/env python3
"""
Raylib performance demo: 5000 ColorBlock widgets in a vertical scroller (10 per row).
Run: python -m selfdrive.ui.colors
"""
import random
import pyray as rl
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.system.ui.widgets.layouts import HBoxLayout

BLOCK_SIZE = 100
BLOCKS_PER_ROW = 10
NUM_BLOCKS = 5000
SPACING = 10


def _random_color() -> rl.Color:
  return rl.Color(
    random.randint(20, 255),
    random.randint(20, 255),
    random.randint(20, 255),
    255,
  )


class ColorBlock(Widget):
  def __init__(self, color: rl.Color):
    super().__init__()
    self._color = color
    self.set_rect(rl.Rectangle(0, 0, BLOCK_SIZE, BLOCK_SIZE))

  def _render(self, _):
    rl.draw_rectangle_rec(self._rect, self._color)


class ColorsMainLayout(Widget):
  """Main layout: vertical Scroller of rows, each row is an HBoxLayout of 10 ColorBlocks."""

  def __init__(self):
    super().__init__()
    self._font = gui_app.font(FontWeight.BOLD)
    row_w = BLOCKS_PER_ROW * BLOCK_SIZE + (BLOCKS_PER_ROW - 1) * SPACING
    rows = [HBoxLayout(widgets=[ColorBlock(_random_color()) for _ in range(BLOCKS_PER_ROW)], spacing=SPACING) for _ in range(NUM_BLOCKS // BLOCKS_PER_ROW)]
    for r in rows:
      r.set_rect(rl.Rectangle(0, 0, row_w, BLOCK_SIZE))
    self._scroller = Scroller(rows, horizontal=False, spacing=SPACING, pad=SPACING, scroll_indicator=True, edge_shadows=False)
    gui_app.push_widget(self)

  def show_event(self):
    super().show_event()
    self._scroller.show_event()

  def hide_event(self):
    super().hide_event()
    self._scroller.hide_event()

  def _render(self, _: rl.Rectangle) -> bool | int | None:
    stats_h = 140  # FPS above, scroller below
    rl.draw_rectangle_rec(rl.Rectangle(self._rect.x, self._rect.y, self._rect.width, stats_h), rl.Color(0, 0, 0, 180))
    rl.draw_text_ex(self._font, f"{rl.get_fps()} FPS", rl.Vector2(self._rect.x + 20, self._rect.y + 20), 56, 0, rl.WHITE)
    rl.draw_text_ex(self._font, f"{(rl.get_frame_time() or 0) * 1000:.2f} ms", rl.Vector2(self._rect.x + 20, self._rect.y + 84), 36, 0, rl.LIGHTGRAY)
    self._scroller.render(rl.Rectangle(self._rect.x, self._rect.y + stats_h, self._rect.width, self._rect.height - stats_h))
    return None


def main():
  gui_app.init_window("Raylib – 5000 widgets")
  ColorsMainLayout()
  for _ in gui_app.render():
    pass


if __name__ == "__main__":
  main()
