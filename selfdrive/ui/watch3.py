#!/usr/bin/env python3
import pyray as rl

from msgq.visionipc import VisionStreamType
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.onroad.cameraview import CameraView


if __name__ == "__main__":
  gui_app.init_window("watch3")
  road_camera_view = CameraView("camerad", VisionStreamType.VISION_STREAM_ROAD)
  driver_camera_view = CameraView("camerad", VisionStreamType.VISION_STREAM_DRIVER)
  wide_road_camera_view = CameraView("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD)
  try:
    for _ in gui_app.render():
      road_camera_view.render(rl.Rectangle(gui_app.width // 4, 0, gui_app.width // 2, gui_app.height // 2))
      driver_camera_view.render(rl.Rectangle(0, gui_app.height // 2, gui_app.width // 2, gui_app.height // 2))
      wide_road_camera_view.render(rl.Rectangle(gui_app.width // 2, gui_app.height // 2, gui_app.width // 2, gui_app.height // 2))
  finally:
    road_camera_view.close()
    driver_camera_view.close()
    wide_road_camera_view.close()
