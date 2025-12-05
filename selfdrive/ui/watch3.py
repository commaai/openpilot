#!/usr/bin/env python3
import os

import pyray as rl

from msgq.visionipc import VisionStreamType
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.onroad.cameraview import CameraView

SERVERS = ["focusing_1", "focusing_2", "focusing_3"]
ROW_COLORS = [rl.ORANGE, rl.BLUE, rl.GREEN]
MAX_SAME = 60

def render_na(rect: rl.Rectangle, font_size: int = 32):
  rl.draw_rectangle_rec(rect, rl.BLACK)

  text = "N/A"
  tw = rl.measure_text(text, font_size)
  th = font_size

  x = int(rect.x + (rect.width - tw) / 2)
  y = int(rect.y + (rect.height - th) / 2)

  rl.draw_text(text, x, y, font_size, rl.WHITE)

if __name__ == "__main__":
  gui_app.init_window("CAMERA FOCUSING")

  roads = [[CameraView(x, VisionStreamType.VISION_STREAM_ROAD), 0, 0] for x in SERVERS]
  drivers = [[CameraView(x, VisionStreamType.VISION_STREAM_DRIVER), 0, 0] for x in SERVERS]
  wides = [[CameraView(x, VisionStreamType.VISION_STREAM_WIDE_ROAD), 0, 0] for x in SERVERS]

  current_device = -1
  cam = None
  zoom = 1.0

  for _ in gui_app.render():
    for cc in (roads, drivers, wides):
      for i in range(len(SERVERS)):
        if cc[i][0].client.frame_id == cc[i][1]:
          cc[i][2] += 1
        else:
          cc[i][2] = 0
        cc[i][1] = cc[i][0].client.frame_id

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
      if zoom == 1.0:
        rect = rl.Rectangle(0, 0, gui_app.width, gui_app.height)
      else:
        dst_w = gui_app.width * zoom
        dst_h = gui_app.height * zoom
        dst_x = (gui_app.width - dst_w) / 2.0
        dst_y = (gui_app.height - dst_h) / 2.0
        rect = rl.Rectangle(dst_x, dst_y, dst_w, dst_h)

      cam[0][0].render(rect)
      if cam[0][2] > MAX_SAME:
        render_na(rect)

      rl.draw_text(cam[1], 10, 10, 80, ROW_COLORS[current_device])
      #rl.draw_text('SHIFT TO ZOOM', 10, 90, 30, ROW_COLORS[current_device])
    elif current_device != -1:
      roads[current_device][0].render(rl.Rectangle(gui_app.width // 4, 0, gui_app.width // 2, gui_app.height // 2))
      rl.draw_text("1", gui_app.width // 2 + gui_app.width // 4 - 100, 0, 80, ROW_COLORS[current_device])
      if roads[current_device][2] > MAX_SAME:
        render_na(rl.Rectangle(gui_app.width // 4, 0, gui_app.width // 2, gui_app.height // 2))

      drivers[current_device][0].render(rl.Rectangle(0, gui_app.height // 2, gui_app.width // 2, gui_app.height // 2))
      rl.draw_text("2", gui_app.height - 120, gui_app.height // 2, 80, ROW_COLORS[current_device])
      if drivers[current_device][2] > MAX_SAME:
        render_na(rl.Rectangle(0, gui_app.height // 2, gui_app.width // 2, gui_app.height // 2))

      wides[current_device][0].render(rl.Rectangle(gui_app.width // 2, gui_app.height // 2, gui_app.width // 2, gui_app.height // 2))
      rl.draw_text("3", gui_app.width - 120, gui_app.height // 2, 80, ROW_COLORS[current_device])
      if wides[current_device][2] > MAX_SAME:
        render_na(rl.Rectangle(gui_app.width // 2, gui_app.height // 2, gui_app.width // 2, gui_app.height // 2))

      rl.draw_text(f"DEVICE {current_device + 1}", 10, 10, 80, ROW_COLORS[current_device])
    else:

      right_panel_frac = 1
      panel_w = int(gui_app.width * right_panel_frac)
      panel_x = gui_app.width - panel_w

      row_h = gui_app.height // 3
      col_w = panel_w // 3

      pad_outer = 8
      pad_inner = 4

      for i in range(len(SERVERS)):
        y = row_h * i

        border_x = panel_x
        border_y = y
        border_w = panel_w
        border_h = row_h

        rl.draw_rectangle_lines_ex(
          rl.Rectangle(border_x, border_y, border_w, border_h),
          10,
          ROW_COLORS[i % len(ROW_COLORS)]
        )


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

        roads[i][0].render(inset_rect(r0.x + 250, r0.y, r0.width, r0.height, pad_inner))
        if roads[i][2] > MAX_SAME:
          render_na(inset_rect(r0.x + 250, r0.y, r0.width, r0.height, pad_inner))

        drivers[i][0].render(inset_rect(r1.x + 150, r1.y, r1.width, r1.height, pad_inner))
        if drivers[i][2] > MAX_SAME:
          render_na(inset_rect(r1.x + 150, r1.y, r1.width, r1.height, pad_inner))

        wides[i][0].render(inset_rect(r2.x + 50, r2.y, r2.width, r2.height, pad_inner))
        if wides[i][2] > MAX_SAME:
          render_na(inset_rect(r2.x + 50, r2.y, r2.width, r2.height, pad_inner))

        label = f"Device {i+1}"

        rl.draw_text(
          label,
          int(panel_x + 50),
          int(y + 150),
          50,
          ROW_COLORS[i % len(ROW_COLORS)]
        )
