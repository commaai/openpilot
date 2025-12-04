#!/usr/bin/env python3
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
  for _ in gui_app.render():
    if rl.is_key_pressed(rl.KEY_ONE):
      cam = (road, 'F')
    elif rl.is_key_pressed(rl.KEY_TWO):
      cam = (driver, 'D')
    elif rl.is_key_pressed(rl.KEY_THREE):
      cam = (wide, 'E')

    rl.draw_text(cam[1], 10, 10, 80, rl.GREEN)
    cam[0].render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
