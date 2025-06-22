import numpy as np
import pyray as rl
from msgq.visionipc import VisionStreamType
from openpilot.selfdrive.ui.onroad.cameraview import CameraView
from openpilot.selfdrive.ui.onroad.driver_state import DriverStateRenderer
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.label import gui_label


class DriverCameraDialog(CameraView):
  def __init__(self):
    super().__init__("camerad", VisionStreamType.VISION_STREAM_DRIVER)
    self.driver_state_renderer = DriverStateRenderer()

  def _render(self, rect):
    super()._render(rect)

    if rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
      return 1

    if not self.frame:
      gui_label(
        rect,
        "camera starting",
        font_size=100,
        font_weight=FontWeight.BOLD,
        alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER,
      )
      return -1

    self._draw_face_detection(rect)
    self.driver_state_renderer.render(rect)

    return -1

  def _draw_face_detection(self, rect: rl.Rectangle) -> None:
    driver_state = ui_state.sm["driverStateV2"]
    is_rhd = driver_state.wheelOnRightProb > 0.5
    driver_data = driver_state.rightDriverData if is_rhd else driver_state.leftDriverData
    face_detect = driver_data.faceProb > 0.7
    if not face_detect:
      return

    # Get face position and orientation
    face_x, face_y = driver_data.facePosition
    face_std = max(driver_data.faceOrientationStd[0], driver_data.faceOrientationStd[1])
    alpha = 0.7
    if face_std > 0.15:
      alpha = max(0.7 - (face_std - 0.15) * 3.5, 0.0)

    # use approx instead of distort_points
    # TODO: replace with distort_points
    fbox_x = int(1080.0 - 1714.0 * face_x)
    fbox_y = int(-135.0 + (504.0 + abs(face_x) * 112.0) + (1205.0 - abs(face_x) * 724.0) * face_y)
    box_size = 220

    line_color = rl.Color(255, 255, 255, int(alpha * 255))
    rl.draw_rectangle_rounded_lines_ex(
      rl.Rectangle(fbox_x - box_size / 2, fbox_y - box_size / 2, box_size, box_size),
      35.0 / box_size / 2,
      10,
      10,
      line_color,
    )

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

  driver_camera_view = DriverCameraDialog()
  try:
    for _ in gui_app.render():
      ui_state.update()
      driver_camera_view.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
  finally:
    driver_camera_view.close()
