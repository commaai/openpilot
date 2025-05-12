#!/usr/bin/env python3
import os

import cairosvg
import pyray as rl
from openpilot.common.basedir import BASEDIR
from openpilot.system.ui.lib.application import gui_app

RENDER_SCALE = 2
ORIGIN = rl.Vector2(0, 0)


class Icon:
  def __init__(self, asset_name: str, origin: rl.Vector2 = ORIGIN):
    self.origin = origin

    data = cairosvg.svg2png(url=os.path.join(BASEDIR, "selfdrive", "assets", asset_name), scale=RENDER_SCALE)
    image = rl.load_image_from_memory(".png", bytes(data), len(data))
    self.width = image.width
    self.height = image.height
    self.texture = rl.load_texture_from_image(image)
    rl.unload_image(image)

  def render(self, pos: rl.Vector2, width: float | None = None, height: float | None = None, scale: float = 1.0, rotation: float = 0.0):
    if width is not None:
      w = width
      h = width * (self.height / self.width)
    elif height is not None:
      h = height
      w = height * (self.width / self.height)
    else:
      w = self.width * scale
      h = self.height * scale
    origin = rl.Vector2(self.origin.x * w, self.origin.y * h)
    rl.draw_texture_pro(self.texture, rl.Rectangle(0, 0, self.width, self.height),
                        rl.Rectangle(pos.x, pos.y, w, h), origin, rotation, rl.WHITE)

  def close(self):
    rl.unload_texture(self.texture)


if __name__ == "__main__":
  gui_app.init_window("Icon")
  icon = Icon("img_circled_check.svg")
  # icon = Icon("img_continue_triangle.svg")
  for _ in gui_app.render():
    icon.render(rl.Vector2(10, 10), scale=1)
  icon.close()
  gui_app.close()
