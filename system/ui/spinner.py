#!/usr/bin/env python3
import pyray as rl
import os
import threading

from openpilot.common.basedir import BASEDIR
from openpilot.system.ui.lib.application import gui_app

# Constants
PROGRESS_BAR_WIDTH = 1000
PROGRESS_BAR_HEIGHT = 20
ROTATION_TIME_SECONDS = 1.0  # Time for one full circle
MARGIN = 200
TEXTURE_SIZE = 360
FONT_SIZE = 80


def clamp(value, min_value, max_value):
  return max(min(value, max_value), min_value)


class Spinner:
  def __init__(self):
    self._comma_texture = gui_app.load_texture_from_image(os.path.join(BASEDIR, "selfdrive/assets/img_spinner_comma.png"), TEXTURE_SIZE, TEXTURE_SIZE)
    self._spinner_texture = gui_app.load_texture_from_image(os.path.join(BASEDIR, "selfdrive/assets/img_spinner_track.png"), TEXTURE_SIZE, TEXTURE_SIZE)
    self._rotation = 0.0
    self._text: str = ""
    self._lock = threading.Lock()

  def set_text(self, text: str) -> None:
    with self._lock:
      self._text = text

  def render(self):
    center = rl.Vector2(gui_app.width / 2.0, gui_app.height / 2.0)
    spinner_origin = rl.Vector2(TEXTURE_SIZE / 2.0, TEXTURE_SIZE / 2.0)
    comma_position = rl.Vector2(center.x - TEXTURE_SIZE / 2.0, center.y - TEXTURE_SIZE / 2.0)

    fps = rl.get_fps()
    if fps > 0:
      degrees_per_frame = 360.0 / (ROTATION_TIME_SECONDS * fps)
      self._rotation = (self._rotation + degrees_per_frame) % 360.0

    # Draw rotating spinner and static comma logo
    rl.draw_texture_pro(self._spinner_texture, rl.Rectangle(0, 0, TEXTURE_SIZE, TEXTURE_SIZE),
                        rl.Rectangle(center.x, center.y, TEXTURE_SIZE, TEXTURE_SIZE),
                        spinner_origin, self._rotation, rl.WHITE)
    rl.draw_texture_v(self._comma_texture, comma_position, rl.WHITE)

    # Display progress bar or text based on user input
    text = None
    with self._lock:
      text = self._text

    if text:
      y_pos = rl.get_screen_height() - MARGIN - PROGRESS_BAR_HEIGHT
      if text.isdigit():
        progress = clamp(int(text), 0, 100)
        bar = rl.Rectangle(center.x - PROGRESS_BAR_WIDTH / 2.0, y_pos, PROGRESS_BAR_WIDTH, PROGRESS_BAR_HEIGHT)
        rl.draw_rectangle_rounded(bar, 0.5, 10, rl.GRAY)

        bar.width *= progress / 100.0
        rl.draw_rectangle_rounded(bar, 0.5, 10, rl.WHITE)
      else:
        text_size = rl.measure_text_ex(gui_app.font(), text, FONT_SIZE, 1.0)
        rl.draw_text_ex(gui_app.font(), text,
                        rl.Vector2(center.x - text_size.x / 2, y_pos), FONT_SIZE, 1.0, rl.WHITE)


if __name__ == "__main__":
  gui_app.init_window("Spinner")
  spinner = Spinner()
  spinner.set_text("Spinner text")
  for _ in gui_app.render():
    spinner.render()
