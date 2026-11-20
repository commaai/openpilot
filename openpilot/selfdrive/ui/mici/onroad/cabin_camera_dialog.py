import math
import raylib_nb as rl
from openpilot.cereal import log, messaging
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.selfdrive.ui.mici.onroad.cameraview import CameraView
from openpilot.selfdrive.ui.mici.onroad.driver_state import DriverStateRenderer
from openpilot.selfdrive.ui.ui_state import ui_state, device
from openpilot.system.ui.lib.application import gui_app, FontWeight, TextAlignment
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.nav_widget import NavWidget
from openpilot.system.ui.widgets.label import gui_label


class CabinCameraView(CameraView):
  def _calc_frame_matrix(self, rect: rl.Rectangle):
    base = super()._calc_frame_matrix(rect)
    driver_view_ratio = 1.5
    base[0, 0] *= driver_view_ratio
    base[1, 1] *= driver_view_ratio
    return base


GREEN = rl.Color(0, 255, 64, 255)
GREY = rl.Color(166, 166, 166, 255)
ORANGE = rl.Color(255, 115, 0, 255)
RED = rl.Color(255, 0, 21, 255)

PANEL_WIDTH = 204
ROW_HEIGHT = 30
MARGIN = 14


def with_alpha(color: rl.Color, alpha: float) -> rl.Color:
  return rl.Color(color.r, color.g, color.b, int(255 * max(0.0, min(1.0, alpha))))


def draw_text(text: str, x: float, y: float, size: int, weight: FontWeight, color: rl.Color):
  rl.draw_text_ex(gui_app.font(weight), text, rl.Vector2(round(x), round(y)), size, 0, color)


class BaseCabinCameraDialog(Widget):
  # Not a NavWidget so training guide can use this without back navigation
  def __init__(self):
    super().__init__()
    self._camera_view = CabinCameraView("camerad", VisionStreamType.VISION_STREAM_CABIN)
    self.driver_state_renderer = DriverStateRenderer()
    self._pm: messaging.PubMaster | None = None

    dt = 1 / gui_app.target_fps
    self._face_x = FirstOrderFilter(0.0, 0.08, dt, initialized=False)
    self._face_y = FirstOrderFilter(0.0, 0.08, dt, initialized=False)
    self._face_alpha = FirstOrderFilter(0.0, 0.1, dt)
    self._awareness = FirstOrderFilter(1.0, 0.15, dt)
    self._signals = {name: FirstOrderFilter(0.0, 0.1, dt) for name in ("face", "eyes", "sunglasses", "phone")}

  def show_event(self):
    super().show_event()
    ui_state.params.put_bool("IsDriverViewEnabled", True)
    self._publish_alert_sound(None)
    device.set_override_interactive_timeout(300)
    ui_state.params.remove("DriverTooDistracted")
    self._pm = messaging.PubMaster(['selfdriveState'])

  def hide_event(self):
    super().hide_event()
    ui_state.params.put_bool("IsDriverViewEnabled", False)
    device.set_override_interactive_timeout(None)

  def _handle_mouse_release(self, _):
    ui_state.params.remove("DriverTooDistracted")

  def __del__(self):
    self.close()

  def close(self):
    if self._camera_view:
      self._camera_view.close()

  def _update_state(self):
    if self._camera_view:
      self._camera_view._update_state()
    # Enable driver state renderer to show Dmoji in preview
    self.driver_state_renderer.set_should_draw(True)
    self.driver_state_renderer.set_force_active(True)
    super()._update_state()

  def _render(self, rect):
    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    self._camera_view._render(rect)

    if not self._camera_view.frame:
      gui_label(rect, tr("camera starting"), font_size=54, font_weight=FontWeight.BOLD,
                alignment=TextAlignment.CENTER)
      rl.end_scissor_mode()
      self._publish_alert_sound(None)
      return

    dm_state = ui_state.sm["driverMonitoringState"]
    driver_data = self.driver_state_renderer.get_driver_data()
    self._publish_alert_sound(dm_state)

    title, color, filled = self._status(dm_state)
    if dm_state.alertLevel.raw >= 2:
      pulse = 0.6 + 0.4 * math.sin(rl.get_time() * (8 if dm_state.alertLevel.raw == 3 else 5))
      rl.draw_rectangle_lines_ex(rect, 6, with_alpha(color, pulse))

    face = self._face_position(rect)
    if face is not None:
      self._draw_face_brackets(*face)

    self._draw_status_pill(rect, title, color, filled)
    self._draw_signals(rect, dm_state, driver_data, color)

    rl.end_scissor_mode()

  @staticmethod
  def _status(dm_state) -> tuple[str, rl.Color, bool]:
    """Title, color, and whether the pill is filled with the color"""
    if not dm_state.visionPolicyState.faceDetected:
      return "looking for driver", GREY, False
    level = dm_state.alertLevel.raw
    if level == 3:
      return "disengage immediately", RED, True
    if level == 2:
      return "pay attention", ORANGE, True
    if level == 1:
      return "pay attention", rl.WHITE, False
    return "driver detected", GREEN, False

  def _publish_alert_sound(self, dm_state):
    """Publish selfdriveState with only alertSound field set"""
    if self._pm is None:
      return

    AudibleAlert = log.SelfdriveState.AudibleAlert
    ALERT_SOUNDS = {
      'one': AudibleAlert.preAlert,
      'two': AudibleAlert.promptDistracted,
      'three': AudibleAlert.warningImmediateUser,
    }
    msg = messaging.new_message('selfdriveState')
    if dm_state is not None:
      msg.selfdriveState.alertSound = ALERT_SOUNDS.get(str(dm_state.alertLevel), AudibleAlert.none)
    self._pm.send('selfdriveState', msg)

  def _face_position(self, rect: rl.Rectangle) -> tuple[float, float, float] | None:
    """Smoothed face center in rect and its confidence, None once faded out"""
    dm_state = ui_state.sm["driverMonitoringState"]
    driver_data = self.driver_state_renderer.get_driver_data()
    if dm_state.visionPolicyState.faceDetected:
      face_x, face_y = driver_data.facePosition
      face_std = max(driver_data.faceOrientationStd[0], driver_data.faceOrientationStd[1])
      self._face_alpha.update(1.0 if face_std <= 0.15 else max(1.0 - (face_std - 0.15) * 5.0, 0.0))

      # use approx instead of distort_points
      # TODO: replace with distort_points
      tici_x = 1080.0 - 1714.0 * face_x
      tici_y = -135.0 + (504.0 + abs(face_x) * 112.0) + (1205.0 - abs(face_x) * 724.0) * face_y

      # Tici coords are relative to center, scale offset and map to rect (from 2160x1080)
      self._face_x.update(rect.x + rect.width / 2 + (tici_x - 1080.0) * 1.25 * rect.width / 2160.0)
      self._face_y.update(rect.y + rect.height / 2 + (tici_y - 540.0) * 1.25 * rect.height / 1080.0)
    else:
      self._face_alpha.update(0.0)

    if not self._face_x.initialized or self._face_alpha.x < 0.01:
      return None
    return self._face_x.x, self._face_y.x, self._face_alpha.x

  def _draw_face_detection(self, rect: rl.Rectangle):
    face = self._face_position(rect)
    if face is None:
      return
    x, y, alpha = face
    box_size = 75
    rl.draw_rectangle_rounded_lines_ex(rl.Rectangle(x - box_size / 2, y - box_size / 2, box_size, box_size),
                                       35.0 / box_size / 2, 3, 3, with_alpha(rl.WHITE, alpha * 0.7))

  @staticmethod
  def _draw_face_brackets(x: float, y: float, alpha: float, size: float = 88, thickness: float = 3):
    half, arm = size / 2, size * 0.22
    color = with_alpha(rl.WHITE, alpha * 0.85)
    for sx in (-1, 1):
      for sy in (-1, 1):
        corner = rl.Vector2(x + sx * half, y + sy * half)
        for end in (rl.Vector2(corner.x - sx * arm, corner.y), rl.Vector2(corner.x, corner.y - sy * arm)):
          rl.draw_line_ex(corner, end, thickness, color)
          rl.draw_circle_v(end, thickness / 2, color)
        rl.draw_circle_v(corner, thickness / 2, color)

  def _draw_status_pill(self, rect: rl.Rectangle, title: str, color: rl.Color, filled: bool):
    # on the driver's side, opposite the signals panel
    height = 48
    font = gui_app.font(FontWeight.BOLD)
    font_size = 26
    while font_size > 20 and measure_text_cached(font, title, font_size).x > rect.width - PANEL_WIDTH - 12 - MARGIN * 2 - 64:
      font_size -= 1
    width = measure_text_cached(font, title, font_size).x + 64
    x = rect.x + rect.width - width - MARGIN if self.driver_state_renderer.is_rhd else rect.x + MARGIN
    y = rect.y + MARGIN

    rl.draw_rectangle_rounded(rl.Rectangle(x, y, width, height), 1.0, 16, with_alpha(color, 0.95) if filled else rl.Color(0, 0, 0, 170))
    rl.draw_circle_v(rl.Vector2(x + 26, y + height / 2), 8, rl.WHITE if filled else color)
    draw_text(title, x + 44, y + (height - font_size) / 2, font_size, FontWeight.BOLD, with_alpha(rl.WHITE, 0.95))

  def _draw_signals(self, rect: rl.Rectangle, dm_state, driver_data, color: rl.Color):
    # (label, value, whether high is good)
    rows = [
      ("face", self._signals["face"].update(float(dm_state.visionPolicyState.faceDetected)), True),
      ("eyes open", self._signals["eyes"].update((driver_data.leftEyeProb + driver_data.rightEyeProb) / 2), True),
      ("sunglasses", self._signals["sunglasses"].update(driver_data.sunglassesProb), False),
      ("phone", self._signals["phone"].update(driver_data.phoneProb), False),
    ]

    height = 20 + ROW_HEIGHT * len(rows) + 52
    x = rect.x + 12 if self.driver_state_renderer.is_rhd else rect.x + rect.width - PANEL_WIDTH - 12
    y = rect.y + 12
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, PANEL_WIDTH, height), 0.18, 12, rl.Color(0, 0, 0, 165))

    row_y = y + 12
    meter_w = 56
    meter_x = x + PANEL_WIDTH - 16 - meter_w
    for label, value, high_good in rows:
      active = value > 0.5
      if high_good:
        meter_color = GREEN if active else ORANGE
      else:
        meter_color = ORANGE if active else rl.WHITE
      draw_text(label, x + 16, row_y + 4, 20, FontWeight.ROMAN, with_alpha(rl.WHITE, 0.85 if active or high_good else 0.5))
      self._draw_meter(rl.Rectangle(meter_x, row_y + 10, meter_w, 6), value, with_alpha(meter_color, 0.9))
      row_y += ROW_HEIGHT

    awareness = self._awareness.update(max(0, dm_state.visionPolicyState.awarenessPercent) / 100)
    row_y += 6
    pct = f"{round(awareness * 100)}%"
    draw_text("awareness", x + 16, row_y, 20, FontWeight.ROMAN, with_alpha(rl.WHITE, 0.85))
    draw_text(pct, x + PANEL_WIDTH - 16 - measure_text_cached(gui_app.font(FontWeight.BOLD), pct, 20).x, row_y, 20,
              FontWeight.BOLD, with_alpha(rl.WHITE, 0.95))
    self._draw_meter(rl.Rectangle(x + 16, row_y + 30, PANEL_WIDTH - 32, 6), awareness, color)

  @staticmethod
  def _draw_meter(rect: rl.Rectangle, value: float, color: rl.Color):
    rl.draw_rectangle_rounded(rect, 1.0, 6, rl.Color(255, 255, 255, 40))
    if value > 0.02:
      rl.draw_rectangle_rounded(rl.Rectangle(rect.x, rect.y, max(rect.height, rect.width * value), rect.height), 1.0, 6, color)


class CabinCameraDialog(NavWidget, BaseCabinCameraDialog):
  def __init__(self):
    super().__init__()
    # TODO: this can grow unbounded, should be given some thought
    device.add_interactive_timeout_callback(gui_app.pop_widget)


if __name__ == "__main__":
  gui_app.init_window("Cabin Camera View (mici)")

  cabin_camera_view = CabinCameraDialog()
  gui_app.push_widget(cabin_camera_view)
  try:
    for _ in gui_app.render():
      ui_state.update()
  finally:
    cabin_camera_view.close()
