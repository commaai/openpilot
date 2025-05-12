#!/usr/bin/env python3
import os

import cairosvg
import pyray as rl
from openpilot.common.basedir import BASEDIR
from openpilot.system.ui.lib.application import gui_app

SVG_RENDER_SCALE = 2
ORIGIN = rl.Vector2(0, 0)


class Icon:
  def __init__(self, asset_name: str, origin: rl.Vector2 = ORIGIN):
    self.origin = origin

    asset_path = os.path.join(BASEDIR, "selfdrive", "assets", asset_name)
    if asset_name.endswith(".svg"):
      data = cairosvg.svg2png(url=asset_path, scale=SVG_RENDER_SCALE)
      image = rl.load_image_from_memory(".png", bytes(data), len(data))
    else:
      image = rl.load_image(asset_path)
    self.width = image.width
    self.height = image.height
    self.texture = rl.load_texture_from_image(image)
    gui_app._textures.append(self.texture)
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


if __name__ == "__main__":
  gui_app.init_window("Icon")
  checkmark = Icon("img_circled_check.svg", origin=rl.Vector2(0.5, 0.5))
  warning = Icon("img_warning_triangle.svg", invert=True)
  png = Icon("offroad/icon_warning.png")
  experimental = Icon("img_experimental.svg")
  experimental_grey = Icon("img_experimental_grey.svg", invert=True)
  try:
    for _ in gui_app.render():
      checkmark.render(rl.Vector2(400, 400))
      warning.render(rl.Vector2(0, 0), height=128)
      png.render(rl.Vector2(128, 0), height=128)
      experimental.render(rl.Vector2(150, 150), scale=0.2)
      experimental_grey.render(rl.Vector2(300, 150), scale=0.2)
  finally:
    gui_app.close()
