#!/usr/bin/env python3
import pyray as rl

from msgq.visionipc import VisionStreamType
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.onroad.cameraview import CameraView
from openpilot.system.ui.widgets.label import gui_label


class TestCameraView(CameraView):
  """Camera view with adjustable brightness parameter for testing"""
  def __init__(self, name: str, stream_type: VisionStreamType, brightness: float, label: str):
    super().__init__(name, stream_type)
    # Override brightness value
    self._brightness_val = rl.ffi.new("float[1]", [brightness])
    self.label = label


if __name__ == "__main__":
  gui_app.init_window("Driver Camera Test Grid - Fine Tuning")

  # Create test configurations
  # Format: (brightness_value, label)
  configs = [
    (1.0, "1.0 (normal)"),
    (1.2, "1.2"),
    (1.3, "1.3"),
    (1.4, "1.4"),
    (1.5, "1.5"),
    (1.6, "1.6"),
    (1.7, "1.7"),
    (1.8, "1.8"),
    (2.0, "2.0"),
  ]

  # Create camera views
  cameras = []
  for brightness, label in configs:
    cam = TestCameraView("camerad", VisionStreamType.VISION_STREAM_DRIVER, brightness, label)
    cameras.append(cam)

  # Layout: 3x3 grid
  cols = 3
  rows = 3
  cell_width = gui_app.width // cols
  cell_height = gui_app.height // rows

  for _ in gui_app.render():
    for idx, cam in enumerate(cameras):
      row = idx // cols
      col = idx % cols
      x = col * cell_width
      y = row * cell_height

      rect = rl.Rectangle(x, y, cell_width, cell_height)
      cam.render(rect)

      # Draw label at top of each cell
      label_rect = rl.Rectangle(x + 5, y + 5, cell_width - 10, 40)
      rl.draw_rectangle_rec(label_rect, rl.Color(0, 0, 0, 180))
      gui_label(label_rect, cam.label, font_size=32, color=rl.WHITE,
                alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER,
                alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE)
