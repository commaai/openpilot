#!/usr/bin/env python3
import os

import pyray as rl

from msgq.visionipc import VisionStreamType
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.onroad.cameraview import CameraView


if __name__ == "__main__":
  gui_app.init_window("watch3")
  road = CameraView("camerad", VisionStreamType.VISION_STREAM_ROAD)
  driver = CameraView("camerad", VisionStreamType.VISION_STREAM_DRIVER)
  wide = CameraView("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD)

  cam = (road, 'F')
  zoom = 1.0

  for _ in gui_app.render():
    if rl.is_key_pressed(rl.KEY_ONE):
      cam = (road, 'F')
      zoom = 1.0
    elif rl.is_key_pressed(rl.KEY_TWO):
      cam = (driver, 'D')
      zoom = 1.0
    elif rl.is_key_pressed(rl.KEY_THREE):
      cam = (wide, 'E')
      zoom = 1.0

    if cam and (rl.is_key_pressed(rl.KEY_LEFT_SHIFT) or rl.is_key_pressed(rl.KEY_RIGHT_SHIFT)):
      zoom = 4.0 if zoom == 1.0 else 1.0

    if rl.is_key_pressed(rl.KEY_ESCAPE) or os.path.exists('/tmp/new_cam'):
      os.system('rm -f /tmp/new_cam')
      cam = None
      zoom = 1.0

    if cam:
      rl.draw_text(cam[1], 10, 10, 80, rl.GREEN)

      if zoom == 1.0:
        rect = rl.Rectangle(0, 0, gui_app.width, gui_app.height)
      else:
        dst_w = gui_app.width * zoom
        dst_h = gui_app.height * zoom
        dst_x = (gui_app.width - dst_w) / 2.0
        dst_y = (gui_app.height - dst_h) / 2.0
        rect = rl.Rectangle(dst_x, dst_y, dst_w, dst_h)

      cam[0].render(rect)
    else:
      road.render(rl.Rectangle(gui_app.width // 4, 0, gui_app.width // 2, gui_app.height // 2))
      driver.render(rl.Rectangle(0, gui_app.height // 2, gui_app.width // 2, gui_app.height // 2))
      wide.render(rl.Rectangle(gui_app.width // 2, gui_app.height // 2, gui_app.width // 2, gui_app.height // 2))
