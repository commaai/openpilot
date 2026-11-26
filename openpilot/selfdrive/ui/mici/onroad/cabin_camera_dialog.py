import math
import numpy as np
import raylib_nb as rl
from opendbc.car.structs import car
from openpilot.cereal import log, messaging
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.mici.onroad import SIDE_PANEL_WIDTH, DIMMED_ICON_ALPHA
from openpilot.selfdrive.ui.mici.onroad.alert_renderer import AlertRenderer
from openpilot.selfdrive.ui.mici.onroad.cameraview import CameraView
from openpilot.selfdrive.ui.mici.onroad.driver_state import DriverStateRenderer, CONE_COLOR_GREEN, CONE_COLOR_ORANGE
from openpilot.selfdrive.ui.onroad.alert_renderer import Alert
from openpilot.selfdrive.ui.ui_state import ui_state, device
from openpilot.system.ui.lib.application import gui_app, FontWeight, TextAlignment
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.nav_widget import NavWidget
from openpilot.system.ui.widgets.label import gui_label, UnifiedLabel

AlertLevel = log.DriverMonitoringState.AlertLevel
AlertSize = log.SelfdriveState.AlertSize
AlertStatus = log.SelfdriveState.AlertStatus
VisualAlert = car.CarControl.HUDControl.VisualAlert

DM_ALERTS = {
  (True, AlertLevel.one): Alert("Pay Attention", "", AlertSize.small, AlertStatus.normal, VisualAlert.none, "driverDistracted1/permanent"),
  (True, AlertLevel.two): Alert("Pay Attention", "Driver Distracted", AlertSize.mid, AlertStatus.userPrompt, VisualAlert.steerRequired,
                                "driverDistracted2/permanent"),
  (True, AlertLevel.three): Alert("DISENGAGE IMMEDIATELY", "Driver Distracted", AlertSize.full, AlertStatus.critical, VisualAlert.steerRequired,
                                  "driverDistracted3/permanent"),
  (False, AlertLevel.one): Alert("Touch Steering Wheel: No Face Detected", "", AlertSize.small, AlertStatus.normal, VisualAlert.steerRequired,
                                 "driverUnresponsive1/permanent"),
  (False, AlertLevel.two): Alert("Touch Steering Wheel", "Driver Unresponsive", AlertSize.mid, AlertStatus.userPrompt, VisualAlert.steerRequired,
                                 "driverUnresponsive2/permanent"),
  (False, AlertLevel.three): Alert("DISENGAGE IMMEDIATELY", "Driver Unresponsive", AlertSize.full, AlertStatus.critical, VisualAlert.steerRequired,
                                   "driverUnresponsive3/permanent"),
}

EYE_THRESHOLD = 0.65
BLINK_OPEN = 0.5
BLINK_THRESHOLD = 0.865
PHONE_THRESHOLD = 0.5

RING_RADIUS = 26
RING_FADE_RADIUS = RING_RADIUS * 0.55
RING_CORNER_RADIUS = 2
CROP_SHIFT = 0.1

STATUS_SHADOW_OFFSET, STATUS_SHADOW_SIGMA, STATUS_SHADOW_ALPHA = 2, 2, 0.5
STATUS_SHADOW_QUANTILES = (-1.0, 0.0, 1.0)
STATUS_SHADOW_ALPHA_PER_LAYER = 1 - (1 - STATUS_SHADOW_ALPHA) ** (1 / len(STATUS_SHADOW_QUANTILES) ** 2)

FACE_FRAME_SIZE = 80
FACE_FRAME_BORDER = 6
FACE_FRAME_RADIUS = 12


def dm_alert() -> Alert | None:
  dm_state = ui_state.sm["driverMonitoringState"]
  return DM_ALERTS.get((dm_state.activePolicy == log.DriverMonitoringState.MonitoringPolicy.vision, dm_state.alertLevel.raw))


def _pack(rgb: tuple[int, int, int], alpha: np.ndarray) -> np.ndarray:
  return (rgb[0] | rgb[1] << 8 | rgb[2] << 16 | (np.clip(alpha, 0, 1) * 255).astype(np.uint32) << 24).astype(np.uint32)


def draw_ring(center: rl.Vector2, fraction: float, rgb: tuple[int, int, int], alpha: float):
  arc_len = 2 * math.pi * RING_RADIUS * min(fraction, 1.0)
  if arc_len < 1:
    return

  s = np.linspace(0, arc_len, max(8, int(arc_len)))
  d = np.minimum(np.minimum(s, arc_len - s), RING_CORNER_RADIUS)
  r_out = RING_RADIUS - RING_CORNER_RADIUS + np.sqrt(RING_CORNER_RADIUS ** 2 - (RING_CORNER_RADIUS - d) ** 2)
  direction = np.c_[np.cos(math.pi / 2 + s / RING_RADIUS), np.sin(math.pi / 2 + s / RING_RADIUS)]
  edge_alpha = alpha * (r_out - RING_FADE_RADIUS) / (RING_RADIUS - RING_FADE_RADIUS)

  rl.cpu.target_raylib()
  for r0, r1, a0, a1 in ((RING_FADE_RADIUS, r_out, 0, edge_alpha), (r_out, r_out + 1, edge_alpha, 0)):
    pts = np.empty((2 * len(s), 2), np.float32)
    pts[0::2] = direction * np.broadcast_to(r0, s.shape)[:, None] + (center.x, center.y)
    pts[1::2] = direction * np.broadcast_to(r1, s.shape)[:, None] + (center.x, center.y)
    colors = np.empty(2 * len(s), np.uint32)
    colors[0::2] = _pack(rgb, np.broadcast_to(a0, s.shape))
    colors[1::2] = _pack(rgb, np.broadcast_to(a1, s.shape))
    rl.cpu.triangles(pts, colors, rl.cpu.TRIANGLE_STRIP)


class CabinCameraView(CameraView):
  def _calc_frame_matrix(self, rect: rl.Rectangle):
    base = super()._calc_frame_matrix(rect)
    driver_view_ratio = 1.5
    base[0, 0] *= driver_view_ratio
    base[1, 1] *= driver_view_ratio
    base[1, 2] = -2 * CROP_SHIFT * base[1, 1]
    return base


class BaseCabinCameraDialog(Widget):
  # Not a NavWidget so training guide can use this without back navigation
  def __init__(self):
    super().__init__()
    self._camera_view = CabinCameraView("camerad", VisionStreamType.VISION_STREAM_CABIN)
    self.driver_state_renderer = DriverStateRenderer(background=False)
    self._alert_renderer = AlertRenderer(dm_alert)
    self._pm: messaging.PubMaster | None = None

    self._eye_icon = gui_app.texture("icons_mici/onroad/driver_monitoring/eye.png", 36, 25)
    self._phone_icon = gui_app.texture("icons_mici/onroad/driver_monitoring/phone.png", 26, 36)

    self._eye_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._phone_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._eye_open_filter = FirstOrderFilter(0.0, 0.05, 1 / gui_app.target_fps)
    self._phone_detected_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._status_alpha_filter = FirstOrderFilter(1.0, 0.1, 1 / gui_app.target_fps)

    self._status_label = UnifiedLabel("", 36, FontWeight.SEMI_BOLD, text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                                      letter_spacing=-0.02, wrap_text=False)

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
    super()._update_state()

    driver_data = self.driver_state_renderer.get_driver_data()
    eye = (driver_data.leftEyeProb + driver_data.rightEyeProb) / 2
    self._eye_filter.update(eye)
    blink = (driver_data.leftBlinkProb + driver_data.rightBlinkProb) / 2
    self._eye_open_filter.update(np.interp(blink, [BLINK_OPEN, BLINK_THRESHOLD], [1.0, 0.0]) if eye > EYE_THRESHOLD else 0.0)
    self._phone_filter.update(driver_data.phoneProb)
    self._phone_detected_filter.update(driver_data.phoneProb > PHONE_THRESHOLD)

    dm_state = ui_state.sm["driverMonitoringState"]
    face_detected = dm_state.visionPolicyState.faceDetected
    self._status_label.set_text(tr("driver detected") if face_detected else tr("looking for driver"))

  def _render(self, rect):
    content_rect = rl.Rectangle(rect.x + SIDE_PANEL_WIDTH, rect.y, rect.width - SIDE_PANEL_WIDTH, rect.height)
    rl.draw_rectangle_rec(rect, rl.BLACK)

    rl.begin_scissor_mode(int(content_rect.x), int(content_rect.y), int(content_rect.width), int(content_rect.height))
    self._camera_view._render(content_rect)

    if not self._camera_view.frame:
      gui_label(content_rect, tr("camera starting"), font_size=54, font_weight=FontWeight.BOLD,
                alignment=TextAlignment.CENTER)
      rl.end_scissor_mode()
      self._publish_alert_sound(None)
      return

    self._draw_face_detection(content_rect)
    alert = dm_alert()
    self._publish_alert_sound(ui_state.sm["driverMonitoringState"])
    status_alpha = self._status_alpha_filter.update(alert is None)
    status_rect = rl.Rectangle(content_rect.x + 20, content_rect.y + content_rect.height - 51, content_rect.width - 40, 36)
    self._status_label.set_text_color(rl.Color(0, 0, 0, round(255 * STATUS_SHADOW_ALPHA_PER_LAYER * status_alpha)))
    for zx in STATUS_SHADOW_QUANTILES:
      for zy in STATUS_SHADOW_QUANTILES:
        self._status_label.render(rl.Rectangle(status_rect.x + zx * STATUS_SHADOW_SIGMA, status_rect.y + STATUS_SHADOW_OFFSET + zy * STATUS_SHADOW_SIGMA,
                                               status_rect.width, status_rect.height))
    self._status_label.set_text_color(rl.Color(255, 255, 255, int(255 * 0.9 * status_alpha)))
    self._status_label.render(status_rect)

    self._alert_renderer.render(content_rect)

    rl.draw_rectangle_rounded_lines_ex(content_rect, 0.2 * 1.02, 10, 50, rl.BLACK)
    rl.end_scissor_mode()

    self._render_left_panel(rl.Rectangle(rect.x, rect.y, SIDE_PANEL_WIDTH, rect.height), alert)

  def _render_left_panel(self, rect: rl.Rectangle, alert: Alert | None):
    center_x = rect.x + rect.width / 2

    dm_size = self.driver_state_renderer.rect.width
    self.driver_state_renderer.set_should_draw(True)
    self.driver_state_renderer.set_show_under_alerts(True)
    self.driver_state_renderer.set_warning(alert is not None and alert.status != AlertStatus.normal and alert.alert_type.startswith("driverDistracted"))
    self.driver_state_renderer.set_position(center_x - dm_size / 2, rect.y + 36 - dm_size / 2)
    self.driver_state_renderer.render()

    eye_center = rl.Vector2(center_x, rect.y + 120)
    draw_ring(eye_center, self._eye_filter.x, CONE_COLOR_GREEN, 1.0)
    eye_alpha = DIMMED_ICON_ALPHA + (0.9 - DIMMED_ICON_ALPHA) * self._eye_open_filter.x
    rl.draw_texture_v(self._eye_icon, rl.Vector2(eye_center.x - self._eye_icon.width / 2, eye_center.y - self._eye_icon.height / 2),
                      rl.Color(255, 255, 255, int(255 * eye_alpha)))

    phone_center = rl.Vector2(center_x, rect.y + 204)
    draw_ring(phone_center, self._phone_filter.x, CONE_COLOR_ORANGE, 1.0)
    phone_alpha = DIMMED_ICON_ALPHA + (0.9 - DIMMED_ICON_ALPHA) * self._phone_detected_filter.x
    rl.draw_texture_v(self._phone_icon, rl.Vector2(phone_center.x - self._phone_icon.width / 2, phone_center.y - self._phone_icon.height / 2),
                      rl.Color(255, 255, 255, int(255 * phone_alpha)))

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

  def _draw_face_detection(self, rect: rl.Rectangle):
    dm_state = ui_state.sm["driverMonitoringState"]
    driver_data = self.driver_state_renderer.get_driver_data()
    if not dm_state.visionPolicyState.faceDetected:
      return

    # Get face position and orientation
    face_x, face_y = driver_data.facePosition
    face_std = max(driver_data.faceOrientationStd[0], driver_data.faceOrientationStd[1])
    alpha = 1.0
    if face_std > 0.15:
      alpha = max(1.0 - (face_std - 0.15) * 5, 0.0)

    # use approx instead of distort_points
    # TODO: replace with distort_points
    tici_x = 1080.0 - 1714.0 * face_x
    tici_y = -135.0 + (504.0 + abs(face_x) * 112.0) + (1205.0 - abs(face_x) * 724.0) * face_y

    # Tici coords are relative to center, scale offset
    offset_x = (tici_x - 1080.0) * 1.25
    offset_y = (tici_y - 540.0) * 1.25

    # Map to mici screen (scale from 2160x1080 to rect dimensions)
    scale_x = rect.width / 2160.0
    scale_y = rect.height / 1080.0
    fbox_x = rect.x + rect.width / 2 + offset_x * scale_x
    fbox_y = rect.y + rect.height / 2 + offset_y * scale_y + self._camera_view._calc_frame_matrix(rect)[1, 2] * rect.height / 2

    inner = FACE_FRAME_SIZE - 2 * FACE_FRAME_BORDER
    rl.draw_rectangle_rounded_lines_ex(rl.Rectangle(fbox_x - inner / 2, fbox_y - inner / 2, inner, inner),
                                       (FACE_FRAME_RADIUS - FACE_FRAME_BORDER) * 2 / inner, 10, FACE_FRAME_BORDER,
                                       rl.Color(255, 255, 255, int(255 * alpha)))


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
