#!/usr/bin/env python3
import re
import os
import xml.etree.ElementTree as ET

import pyray as rl
from openpilot.common.basedir import BASEDIR
from openpilot.system.ui.lib.application import gui_app


RENDER_SCALE = 4.0


def render_svg_to_texture(asset_name: str) -> tuple[rl.Rectangle, rl.RenderTexture]:
  """
  Parse a limited subset of SVG (absolute M/L/C/Z, stroke-only)
  and render directly into a high-DPI RenderTexture via pyray.
  Returns (src_rect, render_texture).
  """
  tree = ET.parse(os.path.join(BASEDIR, "selfdrive", "assets", asset_name))
  root = tree.getroot()

  icon_w = int(float(root.get("width", 0)))
  icon_h = int(float(root.get("height", 0)))
  assert icon_w > 0 and icon_h > 0

  vb = root.get("viewBox", None)
  if vb:
    min_x, min_y, vb_w, vb_h = map(float, vb.split())
  else:
    min_x, min_y, vb_w, vb_h = 0.0, 0.0, icon_w, icon_h

  tex_w = int(icon_w * RENDER_SCALE)
  tex_h = int(icon_h * RENDER_SCALE)

  # TODO: use gui_app to manage texture
  target = rl.load_render_texture(tex_w, tex_h)
  rl.begin_texture_mode(target)
  rl.clear_background(rl.BLANK)

  sx = tex_w / vb_w
  sy = tex_h / vb_h

  def tx(x: float) -> float:
    return (x - min_x) * sx

  def ty(y: float) -> float:
    # raylib Y axis is flipped
    return tex_h - (y - min_y) * sy

  # Minimal path‐data tokenizer: commands & floats
  token_re = re.compile(r"[MLCZmlcz]|-?\d+(?:\.\d+)?")

  for path in root.findall("{http://www.w3.org/2000/svg}path"):
    d = path.get("d")
    if not d:
      continue

    stroke = path.get("stroke", "#000000").lower()
    if stroke.startswith("#"):
      hexc = stroke.lstrip("#")
      if len(hexc) == 3:
        hexc = hexc[0] * 2 + hexc[1] * 2 + hexc[2] * 2
      col = rl.Color(int(hexc[0:2], 16), int(hexc[2:4], 16), int(hexc[4:6], 16), 255)
    else:
      # only "white" supported
      col = rl.Color(255, 255, 255, 255)

    sw = float(path.get("stroke-width", "1")) * RENDER_SCALE
    cap = path.get("stroke-linecap", None)

    tokens = token_re.findall(d)
    i = 0
    cmd = None
    relative = False
    curr_x, curr_y = 0.0, 0.0
    start_x, start_y = 0.0, 0.0

    while i < len(tokens):
      t = tokens[i]
      if t in ("M","m","L","l","C","c","Z","z"):
        cmd = t.upper()
        relative = t.islower()
        i += 1
        if cmd == "Z":
          # close path: draw line curr->start
          p0 = rl.Vector2(tx(curr_x), ty(curr_y))
          p1 = rl.Vector2(tx(start_x), ty(start_y))
          rl.draw_line_ex(p0, p1, sw, col)
          curr_x, curr_y = start_x, start_y
        continue

      # number token, so apply last cmd
      if cmd == "M":
        x = float(t)
        y = float(tokens[i + 1])
        if relative:
          x += curr_x
          y += curr_y
        curr_x, curr_y = x, y
        start_x, start_y = x, y
        i += 2
        # subsequent implicit L
        cmd = "L"
      elif cmd == "L":
        x = float(t)
        y = float(tokens[i + 1])
        if relative:
          x += curr_x
          y += curr_y
        p0 = rl.Vector2(tx(curr_x), ty(curr_y))
        p1 = rl.Vector2(tx(x), ty(y))
        if cap == "round":
          rl.draw_circle_v(p0, sw / 2, col)
        rl.draw_line_ex(p0, p1, sw, col)
        curr_x, curr_y = x, y
        i += 2
      elif cmd == "C":
        c1x = float(t)
        c1y = float(tokens[i + 1])
        c2x = float(tokens[i + 2])
        c2y = float(tokens[i + 3])
        x = float(tokens[i + 4])
        y = float(tokens[i + 5])
        if relative:
          c1x += curr_x
          c1y += curr_y
          c2x += curr_x
          c2y += curr_y
          x += curr_x
          y += curr_y
        p0 = rl.Vector2(tx(curr_x), ty(curr_y))
        cp1 = rl.Vector2(tx(c1x), ty(c1y))
        cp2 = rl.Vector2(tx(c2x), ty(c2y))
        p1 = rl.Vector2(tx(x), ty(y))
        rl.draw_spline_segment_bezier_cubic(p0, cp1, cp2, p1, sw, col)
        curr_x, curr_y = x, y
        i += 6
      else:
        # unknown or unsupported
        assert False, f"unknown token {t}"

    if cap == "round":
      rl.draw_circle_v(rl.Vector2(tx(curr_x), ty(curr_y)), sw / 2, col)

  rl.end_texture_mode()
  return rl.Rectangle(0, 0, tex_w, tex_h), target


ORIGIN = rl.Vector2(0.5, 0.5)


class Icon:
  def __init__(self, asset_name: str, origin: rl.Vector2 = ORIGIN):
    self.src_rect, self.texture = render_svg_to_texture(asset_name)
    self.origin = origin

  @property
  def height(self) -> float:
    return self.src_rect.height

  @property
  def width(self) -> float:
    return self.src_rect.width

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
    rl.draw_texture_pro(self.texture.texture, self.src_rect, rl.Rectangle(pos.x, pos.y, w, h), origin, rotation, rl.WHITE)

  def close(self):
    rl.unload_render_texture(self.texture)


if __name__ == "__main__":
  gui_app.init_window("Icon")
  icon = Icon("img_circled_check.svg")
  # icon = Icon("img_continue_triangle.svg")
  for _ in gui_app.render():
    icon.render(rl.Vector2(10, 10), scale=1)
  icon.close()
  gui_app.close()
