#!/usr/bin/env python3
import os

import pyray as rl

from msgq.visionipc import VisionStreamType
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.onroad.cameraview import CameraView

SERVERS = ["focusing_1", "focusing_2", "focusing_3"]
ROW_COLORS = [rl.GREEN, rl.ORANGE, rl.BLUE]

if __name__ == "__main__":
  gui_app.init_window("CAMERA FOCUSING")

  roads = [CameraView(x, VisionStreamType.VISION_STREAM_ROAD) for x in SERVERS]
  drivers = [CameraView(x, VisionStreamType.VISION_STREAM_DRIVER) for x in SERVERS]
  wides = [CameraView(x, VisionStreamType.VISION_STREAM_WIDE_ROAD) for x in SERVERS]

  current_device = -1
  cam = None
  zoom = 1.0

  for _ in gui_app.render():
    if rl.is_key_pressed(rl.KEY_ONE):
      if current_device == -1:
        current_device = 0
      else:
        cam = (roads[current_device], 'F')
        zoom = 1.0
    elif rl.is_key_pressed(rl.KEY_TWO):
      if current_device == -1:
        current_device = 1
      else:
        cam = (drivers[current_device], 'D')
        zoom = 1.0
    elif rl.is_key_pressed(rl.KEY_THREE):
      if current_device == -1:
        current_device = 2
      else:
        cam = (wides[current_device], 'E')
        zoom = 1.0

    if cam and (rl.is_key_pressed(rl.KEY_LEFT_SHIFT) or rl.is_key_pressed(rl.KEY_RIGHT_SHIFT)):
      zoom = 4.0 if zoom == 1.0 else 1.0

    if rl.is_key_pressed(rl.KEY_ESCAPE):
      if cam:
        cam = None
      elif current_device != -1:
        current_device = -1
      zoom = 1.0

    if cam:
      rl.draw_text(cam[1], 10, 10, 80, ROW_COLORS[current_device])

      if zoom == 1.0:
        rect = rl.Rectangle(0, 0, gui_app.width, gui_app.height)
      else:
        dst_w = gui_app.width * zoom
        dst_h = gui_app.height * zoom
        dst_x = (gui_app.width - dst_w) / 2.0
        dst_y = (gui_app.height - dst_h) / 2.0
        rect = rl.Rectangle(dst_x, dst_y, dst_w, dst_h)

      cam[0].render(rect)
    elif current_device != -1:
      rl.draw_text(f"DEVICE {current_device + 1}", 10, 10, 80, ROW_COLORS[current_device])
      roads[current_device].render(rl.Rectangle(gui_app.width // 4, 0, gui_app.width // 2, gui_app.height // 2))
      drivers[current_device].render(rl.Rectangle(0, gui_app.height // 2, gui_app.width // 2, gui_app.height // 2))
      wides[current_device].render(rl.Rectangle(gui_app.width // 2, gui_app.height // 2, gui_app.width // 2, gui_app.height // 2))
    else:

      right_panel_frac = 1  # 45% of screen width for the 3x3 grid
      panel_w = int(gui_app.width * right_panel_frac)
      panel_x = gui_app.width - panel_w

      row_h = gui_app.height // 3
      col_w = panel_w // 3

      pad_outer = 8   # padding inside row border
      pad_inner = 4   # padding between cameras (lightly handled by shrinking rects)

      for i in range(len(SERVERS)):
        y = row_h * i

        # Border rect around ALL 3 cams for this server row
        border_x = panel_x
        border_y = y
        border_w = panel_w
        border_h = row_h

        # Draw an outline (use draw_rectangle_lines_ex for thickness)
        rl.draw_rectangle_lines_ex(
          rl.Rectangle(border_x, border_y, border_w, border_h),
          10,
          ROW_COLORS[i % len(ROW_COLORS)]
        )

        # Compute inner area for cameras (so they sit inside the border nicely)
        inner_x = border_x + pad_outer
        inner_y = border_y + pad_outer
        inner_w = border_w - 2 * pad_outer
        inner_h = border_h - 2 * pad_outer

        cam_w = inner_w / 3.0

        def inset_rect(x, y, w, h, p):
          return rl.Rectangle(x + p, y + p, w - 2*p, h - 2*p)

        r0 = rl.Rectangle(inner_x + cam_w * 0, inner_y, cam_w, inner_h)
        r1 = rl.Rectangle(inner_x + cam_w * 1, inner_y, cam_w, inner_h)
        r2 = rl.Rectangle(inner_x + cam_w * 2, inner_y, cam_w, inner_h)

        roads[i].render(inset_rect(r0.x, r0.y, r0.width, r0.height, pad_inner))
        drivers[i].render(inset_rect(r1.x, r1.y, r1.width, r1.height, pad_inner))
        wides[i].render(inset_rect(r2.x, r2.y, r2.width, r2.height, pad_inner))
