#!/usr/bin/env python3
import pyray as rl
import os
import select
import sys

from openpilot.common.basedir import BASEDIR
from openpilot.system.ui.lib.application import gui_app

# Constants
PROGRESS_BAR_WIDTH = 1000
PROGRESS_BAR_HEIGHT = 20
ROTATION_RATE = 12.0
MARGIN = 200
TEXTURE_SIZE = 360
FONT_SIZE = 80


def clamp(value, min_value, max_value):
  return max(min(value, max_value), min_value)


def check_input_non_blocking():
  if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
    return sys.stdin.readline().strip()
  return ""


def main():
  gui_app.init_window("Spinner")

  # Load textures
  comma_texture = gui_app.load_texture_from_image(os.path.join(BASEDIR, "selfdrive/assets/img_spinner_comma.png"), TEXTURE_SIZE, TEXTURE_SIZE)
  spinner_texture = gui_app.load_texture_from_image(os.path.join(BASEDIR, "selfdrive/assets/img_spinner_track.png"), TEXTURE_SIZE, TEXTURE_SIZE)

  # Initial values
  rotation = 0.0
  user_input = ""
  center = rl.Vector2(gui_app.width / 2.0, gui_app.height / 2.0)
  spinner_origin = rl.Vector2(TEXTURE_SIZE / 2.0, TEXTURE_SIZE / 2.0)
  comma_position = rl.Vector2(center.x - TEXTURE_SIZE / 2.0, center.y - TEXTURE_SIZE / 2.0)

  for _ in gui_app.render():
    # Update rotation
    rotation = (rotation + ROTATION_RATE) % 360.0

    # Draw rotating spinner and static comma logo
    rl.draw_texture_pro(spinner_texture, rl.Rectangle(0, 0, TEXTURE_SIZE, TEXTURE_SIZE),
                        rl.Rectangle(center.x, center.y, TEXTURE_SIZE, TEXTURE_SIZE),
                        spinner_origin, rotation, rl.WHITE)
    rl.draw_texture_v(comma_texture, comma_position, rl.WHITE)

    # Read user input
    if input_str := check_input_non_blocking():
      user_input = input_str

    # Display progress bar or text based on user input
    if user_input:
      y_pos = rl.get_screen_height() - MARGIN - PROGRESS_BAR_HEIGHT
      if user_input.isdigit():
        progress = clamp(int(user_input), 0, 100)
        bar = rl.Rectangle(center.x - PROGRESS_BAR_WIDTH / 2.0, y_pos, PROGRESS_BAR_WIDTH, PROGRESS_BAR_HEIGHT)
        rl.draw_rectangle_rounded(bar, 0.5, 10, rl.GRAY)

        bar.width *= progress / 100.0
        rl.draw_rectangle_rounded(bar, 0.5, 10, rl.WHITE)
      else:
        text_size = rl.measure_text_ex(gui_app.font(), user_input, FONT_SIZE, 1.0)
        rl.draw_text_ex(gui_app.font(), user_input,
                        rl.Vector2(center.x - text_size.x / 2, y_pos), FONT_SIZE, 1.0, rl.WHITE)


if __name__ == "__main__":
  main()
