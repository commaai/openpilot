import pyray as rl
from cereal import log, messaging
from msgq.visionipc import VisionStreamType
from openpilot.selfdrive.ui.mici.onroad.cameraview import CameraView
from openpilot.selfdrive.ui.mici.onroad.driver_state import DriverStateRenderer
from openpilot.selfdrive.ui.ui_state import ui_state, device
from openpilot.selfdrive.selfdrived.events import EVENTS, ET
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import NavWidget
from openpilot.system.ui.widgets.label import gui_label

EventName = log.OnroadEvent.EventName

EVENT_TO_INT = EventName.schema.enumerants


class DriverCameraView(CameraView):
  def _calc_frame_matrix(self, rect: rl.Rectangle):
    base = super()._calc_frame_matrix(rect)
    driver_view_ratio = 1.5
    base[0, 0] *= driver_view_ratio
    base[1, 1] *= driver_view_ratio
    return base


class DriverCameraDialog(NavWidget):
  def __init__(self, no_escape=False):
    super().__init__()
    self._camera_view = DriverCameraView("camerad", VisionStreamType.VISION_STREAM_DRIVER)
    self.driver_state_renderer = DriverStateRenderer(lines=True)
    self.driver_state_renderer.set_rect(rl.Rectangle(0, 0, 200, 200))
    self.driver_state_renderer.load_icons()
    self._pm: messaging.PubMaster | None = None
    if not no_escape:
      # TODO: this can grow unbounded, should be given some thought
      device.add_interactive_timeout_callback(lambda: gui_app.set_modal_overlay(None))
    self.set_back_callback(lambda: gui_app.set_modal_overlay(None))
    self.set_back_enabled(not no_escape)

    # Load eye icons
    self._eye_fill_texture = None
    self._eye_orange_texture = None
    self._eye_size = 74
    self._glasses_texture = None
    self._glasses_size = 171

    self._load_eye_textures()

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
                alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)
      rl.end_scissor_mode()
      self._publish_alert_sound(None)
      return -1

    driver_data = self._draw_face_detection(rect)
    if driver_data is not None:
      self._draw_eyes(rect, driver_data)

    # Position dmoji on opposite side from driver
    driver_state_rect = (
      rect.x if self.driver_state_renderer.is_rhd else rect.x + rect.width - self.driver_state_renderer.rect.width,
      rect.y + (rect.height - self.driver_state_renderer.rect.height) / 2,
    )
    self.driver_state_renderer.set_position(*driver_state_rect)
    self.driver_state_renderer.render()

    # Render driver monitoring alerts
    self._render_dm_alerts(rect)

    rl.end_scissor_mode()
    return -1

  def _publish_alert_sound(self, dm_state):
    """Publish selfdriveState with only alertSound field set"""
    if self._pm is None:
      return

    msg = messaging.new_message('selfdriveState')
    if dm_state is not None and len(dm_state.events):
      event_name = EVENT_TO_INT[dm_state.events[0].name]
      if event_name is not None and event_name in EVENTS and ET.PERMANENT in EVENTS[event_name]:
        msg.selfdriveState.alertSound = EVENTS[event_name][ET.PERMANENT].audible_alert
    self._pm.send('selfdriveState', msg)

  def _render_dm_alerts(self, rect: rl.Rectangle):
    """Render driver monitoring event names"""
    dm_state = ui_state.sm["driverMonitoringState"]
    self._publish_alert_sound(dm_state)

    gui_label(rl.Rectangle(rect.x + 2, rect.y + 2, rect.width, rect.height),
              f"Awareness: {dm_state.awarenessStatus * 100:.0f}%", font_size=44, font_weight=FontWeight.MEDIUM,
              alignment=rl.GuiTextAlignment.TEXT_ALIGN_RIGHT,
              alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP,
              color=rl.Color(0, 0, 0, 180))
    gui_label(rect, f"Awareness: {dm_state.awarenessStatus * 100:.0f}%", font_size=44, font_weight=FontWeight.MEDIUM,
              alignment=rl.GuiTextAlignment.TEXT_ALIGN_RIGHT,
              alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP,
              color=rl.Color(255, 255, 255, int(255 * 0.9)))

    if not dm_state.events:
      return

    # Show first event (only one should be active at a time)
    event_name_str = str(dm_state.events[0].name).split('.')[-1]
    alignment = rl.GuiTextAlignment.TEXT_ALIGN_RIGHT if self.driver_state_renderer.is_rhd else rl.GuiTextAlignment.TEXT_ALIGN_LEFT

    shadow_rect = rl.Rectangle(rect.x + 2, rect.y + 2, rect.width, rect.height)
    gui_label(shadow_rect, event_name_str, font_size=40, font_weight=FontWeight.BOLD,
              alignment=alignment,
              alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_BOTTOM,
              color=rl.Color(0, 0, 0, 180))
    gui_label(rect, event_name_str, font_size=40, font_weight=FontWeight.BOLD,
              alignment=alignment,
              alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_BOTTOM,
              color=rl.Color(255, 255, 255, int(255 * 0.9)))

  def _load_eye_textures(self):
    """Lazy load eye textures"""
    if self._eye_fill_texture is None:
      self._eye_fill_texture = gui_app.texture("icons_mici/onroad/eye_fill.png", self._eye_size, self._eye_size)
    if self._eye_orange_texture is None:
      self._eye_orange_texture = gui_app.texture("icons_mici/onroad/eye_orange.png", self._eye_size, self._eye_size)
    if self._glasses_texture is None:
      self._glasses_texture = gui_app.texture("icons_mici/onroad/glasses.png", self._glasses_size, self._glasses_size)

  def _draw_face_detection(self, rect: rl.Rectangle):
    dm_state = ui_state.sm["driverMonitoringState"]
    driver_data = self.driver_state_renderer.get_driver_data()
    if not dm_state.faceDetected:
      return

    # Get face position and orientation
    face_x, face_y = driver_data.facePosition
    face_std = max(driver_data.faceOrientationStd[0], driver_data.faceOrientationStd[1])
    alpha = 0.7
    if face_std > 0.15:
      alpha = max(0.7 - (face_std - 0.15) * 3.5, 0.0)

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
    fbox_y = rect.y + rect.height / 2 + offset_y * scale_y
    box_size = 75
    line_thickness = 3

    line_color = rl.Color(255, 255, 255, int(alpha * 255))
    rl.draw_rectangle_rounded_lines_ex(
      rl.Rectangle(fbox_x - box_size / 2, fbox_y - box_size / 2, box_size, box_size),
      35.0 / box_size / 2,
      line_thickness,
      line_thickness,
      line_color,
    )
    return driver_data

  def _draw_eyes(self, rect: rl.Rectangle, driver_data):
    # Draw eye indicators based on eye probabilities
    eye_offset_x = 10
    eye_offset_y = 10
    eye_spacing = self._eye_size + 15

    left_eye_x = rect.x + eye_offset_x
    left_eye_y = rect.y + eye_offset_y
    left_eye_prob = driver_data.leftEyeProb

    right_eye_x = rect.x + eye_offset_x + eye_spacing
    right_eye_y = rect.y + eye_offset_y
    right_eye_prob = driver_data.rightEyeProb

    # Draw eyes with opacity based on probability
    for eye_x, eye_y, eye_prob in [(left_eye_x, left_eye_y, left_eye_prob), (right_eye_x, right_eye_y, right_eye_prob)]:
      fill_opacity = eye_prob
      orange_opacity = 1.0 - eye_prob

      rl.draw_texture_v(self._eye_orange_texture, (eye_x, eye_y), rl.Color(255, 255, 255, int(255 * orange_opacity)))
      rl.draw_texture_v(self._eye_fill_texture, (eye_x, eye_y), rl.Color(255, 255, 255, int(255 * fill_opacity)))

    # Draw sunglasses indicator based on sunglasses probability
    # Position glasses centered between the two eyes at top left
    glasses_x = rect.x + eye_offset_x - 4
    glasses_y = rect.y
    glasses_pos = rl.Vector2(glasses_x, glasses_y)
    glasses_prob = driver_data.sunglassesProb
    rl.draw_texture_v(self._glasses_texture, glasses_pos, rl.Color(70, 80, 161, int(255 * glasses_prob)))


if __name__ == "__main__":
  gui_app.init_window("Driver Camera View (mici)")

  driver_camera_view = DriverCameraDialog()
  try:
    for _ in gui_app.render():
      ui_state.update()
      driver_camera_view.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
  finally:
    driver_camera_view.close()
