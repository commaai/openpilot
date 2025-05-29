import numpy as np
import pyray as rl
from openpilot.system.ui.widgets.cameraview import CameraView
from msgq.visionipc import VisionStreamType
from openpilot.system.ui.lib.application import gui_app


class DriverCameraView(CameraView):
  def __init__(self, stream_type: VisionStreamType):
    super().__init__("camerad", stream_type)

  def render(self, rect):
    super().render(rect)

    # TODO: Add additional rendering logic

  def _calc_frame_matrix(self, rect: rl.Rectangle) -> np.ndarray:
    driver_view_ratio = 2.0

    # Get stream dimensions
    if self.frame:
      stream_width = self.frame.width
      stream_height = self.frame.height
    else:
      # Default values if frame not available
      stream_width = 1928
      stream_height = 1208

    yscale = stream_height * driver_view_ratio / stream_width
    xscale = yscale * rect.height / rect.width * stream_width / stream_height

    return np.array([
      [xscale, 0.0, 0.0],
      [0.0, yscale, 0.0],
      [0.0, 0.0, 1.0]
    ])


if __name__ == "__main__":
  gui_app.init_window("Driver Camera View")
  driver_camera_view = DriverCameraView(VisionStreamType.VISION_STREAM_DRIVER)
  try:
    for _ in gui_app.render():
      driver_camera_view.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
  finally:
    driver_camera_view.close()
