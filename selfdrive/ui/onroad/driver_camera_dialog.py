import numpy as np
import pyray as rl
from collections.abc import Callable
from msgq.visionipc import VisionStreamType
from openpilot.selfdrive.ui.onroad.cameraview import CameraView
from openpilot.selfdrive.ui.onroad.driver_state import DriverStateRenderer
from openpilot.selfdrive.ui.ui_state import UI_BORDER_SIZE, ui_state, device
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets.button import Button, ButtonStyle
from openpilot.system.ui.widgets.label import gui_label

CLOSE_BTN_SIZE = 192
BUTTON_BORDER_SIZE = 100
BUTTON_WIDTH = 440
BUTTON_HEIGHT = 192
FONT_SIZE = 70

DEFAULT_STREAM_INDEX = 0
STREAM_OPTIONS = [
  VisionStreamType.VISION_STREAM_DRIVER,
  VisionStreamType.VISION_STREAM_ROAD,
  VisionStreamType.VISION_STREAM_WIDE_ROAD,
]

class DriverCameraDialog(CameraView):
  def __init__(self, on_close: Callable[[], None] | None = None):
    super().__init__("camerad", STREAM_OPTIONS[DEFAULT_STREAM_INDEX])
    self.driver_state_renderer = DriverStateRenderer()
    self.on_close = on_close

    # TODO: this can grow unbounded, should be given some thought
    device.add_interactive_timeout_callback(self.close)
    ui_state.params.put_bool("IsDriverViewEnabled", True)

    self.current_stream_index = DEFAULT_STREAM_INDEX

    self.stream_switch_button = Button(
      text=self._get_stream_button_text,
      click_callback=self._rotate_stream,
      font_size=FONT_SIZE,
      font_weight=FontWeight.BOLD,
      button_style=ButtonStyle.PRIMARY,
      border_radius=BUTTON_BORDER_SIZE,
    )

    self.close_button = Button(
      "",
      icon=gui_app.texture("icons/close2.png", FONT_SIZE, FONT_SIZE),
      click_callback=self.close,
      button_style=ButtonStyle.DANGER,
      border_radius=BUTTON_BORDER_SIZE,
    )

  def _get_stream_button_text(self) -> str:
    return {
      VisionStreamType.VISION_STREAM_ROAD: tr("Road"),
      VisionStreamType.VISION_STREAM_WIDE_ROAD: tr("Wide"),
      VisionStreamType.VISION_STREAM_DRIVER: tr("Driver")
    }.get(STREAM_OPTIONS[self.current_stream_index], tr("Unknown"))

  def _handle_mouse_release(self, _):
    super()._handle_mouse_release(_)

  def _rotate_stream(self):
    self.current_stream_index = (self.current_stream_index + 1) % len(STREAM_OPTIONS)
    self.stream_switch_button.set_text(self._get_stream_button_text())
    self.switch_stream(STREAM_OPTIONS[self.current_stream_index])

  def close(self):
    super().close()

    ui_state.params.put_bool("IsDriverViewEnabled", False)
    gui_app.set_modal_overlay(None)

    if self.on_close:
      self.on_close()

  def _render(self, rect):
    super()._render(rect)

    if not self.frame:
      gui_label(
        rect,
        tr("camera starting"),
        font_size=FONT_SIZE,
        font_weight=FontWeight.BOLD,
        alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER,
      )
      return -1

    if self.stream_type == VisionStreamType.VISION_STREAM_DRIVER:
      self._draw_face_detection(rect)
      self.driver_state_renderer.render(rect)

    close_button_rect = rl.Rectangle(UI_BORDER_SIZE, UI_BORDER_SIZE, CLOSE_BTN_SIZE, CLOSE_BTN_SIZE)
    self.close_button.render(close_button_rect)

    button_x = rect.x + rect.width - BUTTON_WIDTH - UI_BORDER_SIZE
    button_y = rect.y + rect.height - BUTTON_HEIGHT - UI_BORDER_SIZE
    button_rect = rl.Rectangle(button_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)
    self.stream_switch_button.render(button_rect)

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
  import sys
  gui_app.init_window("Driver Camera View")

  driver_camera_view = DriverCameraDialog(on_close=lambda: sys.exit(0))
  try:
    for _ in gui_app.render():
      ui_state.update()
      driver_camera_view.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
  finally:
    driver_camera_view.close()
