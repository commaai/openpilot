#!/usr/bin/env python3
import pyray as rl
import os
import threading

from openpilot.common.basedir import BASEDIR
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.text import wrap_text

# Constants
PROGRESS_BAR_WIDTH = 1000
PROGRESS_BAR_HEIGHT = 20
DEGREES_PER_SECOND = 360.0  # one full rotation per second
MARGIN_H = 100
MARGIN_V = 200
TEXTURE_SIZE = 360
FONT_SIZE = 88
LINE_HEIGHT = 96
DARKGRAY = (55, 55, 55, 255)


def clamp(value, min_value, max_value):
  return max(min(value, max_value), min_value)


class Spinner:
  def __init__(self):
    self._comma_texture = gui_app.load_texture_from_image(os.path.join(BASEDIR, "selfdrive/assets/img_spinner_comma.png"), TEXTURE_SIZE, TEXTURE_SIZE)
    self._spinner_texture = gui_app.load_texture_from_image(os.path.join(BASEDIR, "selfdrive/assets/img_spinner_track.png"), TEXTURE_SIZE, TEXTURE_SIZE,
                                                            alpha_premultiply=True)
    self._rotation = 0.0
    self._progress: int | None = None
    self._wrapped_lines: list[str] = []
    self._lock = threading.Lock()

  def set_text(self, text: str) -> None:
    with self._lock:
      if text.isdigit():
        self._progress = clamp(int(text), 0, 100)
        self._wrapped_lines = []
      else:
        self._progress = None
        self._wrapped_lines = wrap_text(text, FONT_SIZE, gui_app.width - MARGIN_H)

  def render(self):
    center = rl.Vector2(gui_app.width / 2.0, gui_app.height / 2.0)
    spinner_origin = rl.Vector2(TEXTURE_SIZE / 2.0, TEXTURE_SIZE / 2.0)
    comma_position = rl.Vector2(center.x - TEXTURE_SIZE / 2.0, center.y - TEXTURE_SIZE / 2.0)

    delta_time = rl.get_frame_time()
    self._rotation = (self._rotation + DEGREES_PER_SECOND * delta_time) % 360.0

    # Draw rotating spinner and static comma logo
    rl.draw_texture_pro(self._spinner_texture, rl.Rectangle(0, 0, TEXTURE_SIZE, TEXTURE_SIZE),
                        rl.Rectangle(center.x, center.y, TEXTURE_SIZE, TEXTURE_SIZE),
                        spinner_origin, self._rotation, rl.WHITE)
    rl.draw_texture_v(self._comma_texture, comma_position, rl.WHITE)

    # Display progress bar or text based on user input
    y_pos = rl.get_screen_height() - MARGIN_V - PROGRESS_BAR_HEIGHT
    with self._lock:
      progress = self._progress
      wrapped_lines = self._wrapped_lines

    if progress is not None:
      bar = rl.Rectangle(center.x - PROGRESS_BAR_WIDTH / 2.0, y_pos, PROGRESS_BAR_WIDTH, PROGRESS_BAR_HEIGHT)
      rl.draw_rectangle_rounded(bar, 1, 10, DARKGRAY)

      bar.width *= progress / 100.0
      rl.draw_rectangle_rounded(bar, 1, 10, rl.WHITE)
    elif wrapped_lines:
      for i, line in enumerate(wrapped_lines):
        text_size = rl.measure_text_ex(gui_app.font(), line, FONT_SIZE, 0.0)
        rl.draw_text_ex(gui_app.font(), line, rl.Vector2(center.x - text_size.x / 2, y_pos + i * LINE_HEIGHT),
                        FONT_SIZE, 0.0, rl.WHITE)


if __name__ == "__main__":
  gui_app.init_window("Spinner")
  spinner = Spinner()
  spinner.set_text("Spinner text")
  for _ in gui_app.render():
    spinner.render()
