import pyray as rl
import numpy as np
import math
from cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget
from openpilot.selfdrive.ui.ui_state import ui_state

AlertSize = log.SelfdriveState.AlertSize

DEBUG = False

LOOKING_CENTER_THRESHOLD_UPPER = math.radians(6)
LOOKING_CENTER_THRESHOLD_LOWER = math.radians(3)


class DriverStateRenderer(Widget):
  BASE_SIZE = 60
  LINES_ANGLE_INCREMENT = 5
  LINES_STALE_ANGLES = 3.0  # seconds

  def __init__(self, lines: bool = False, inset: bool = False):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, self.BASE_SIZE, self.BASE_SIZE))
    self._lines = lines
    self._inset = inset

    # In line mode, track smoothed angles
    assert 360 % self.LINES_ANGLE_INCREMENT == 0
    self._head_angles = {i * self.LINES_ANGLE_INCREMENT: FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps) for i in range(360 // self.LINES_ANGLE_INCREMENT)}

    self._is_active = False
    self._is_rhd = False
    self._face_detected = False
    self._should_draw = False
    self._force_active = False
    self._looking_center = False

    self._fade_filter = FirstOrderFilter(0.0, 0.05, 1 / gui_app.target_fps)
    self._pitch_filter = FirstOrderFilter(0.0, 0.05, 1 / gui_app.target_fps, initialized=False)
    self._yaw_filter = FirstOrderFilter(0.0, 0.05, 1 / gui_app.target_fps, initialized=False)
    self._rotation_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps, initialized=False)
    self._looking_center_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)

    # Load the driver face icons
    self.load_icons()

  def load_icons(self):
    """Load or reload the driver face icon texture"""
    cone_and_person_size = round(52 / self.BASE_SIZE * self._rect.width)

    # If inset is enabled, push cone and person smaller by 2x the current inset space
    if self._inset:
      # Current inset space = (rect.width - cone_and_person_size) / 2
      current_inset = (self._rect.width - cone_and_person_size) / 2
      # Reduce size by 2x the current inset (1x on each side)
      cone_and_person_size = round(cone_and_person_size - current_inset * 2)

    self._dm_person = gui_app.texture("icons_mici/onroad/driver_monitoring/dm_person.png", cone_and_person_size, cone_and_person_size)
    self._dm_cone = gui_app.texture("icons_mici/onroad/driver_monitoring/dm_cone.png", cone_and_person_size, cone_and_person_size)
    center_size = round(36 / self.BASE_SIZE * self._rect.width)
    self._dm_center = gui_app.texture("icons_mici/onroad/driver_monitoring/dm_center.png", center_size, center_size)
    self._dm_background = gui_app.texture("icons_mici/onroad/driver_monitoring/dm_background.png", self._rect.width, self._rect.height)

  def set_should_draw(self, should_draw: bool):
    self._should_draw = should_draw

  @property
  def should_draw(self):
    return (self._should_draw and ui_state.sm["selfdriveState"].alertSize == AlertSize.none and
            ui_state.sm.recv_frame["driverStateV2"] > ui_state.started_frame)

  def set_force_active(self, force_active: bool):
    """Force the dmoji to always appear active (green) regardless of actual state"""
    self._force_active = force_active

  @property
  def effective_active(self) -> bool:
    """Returns True if dmoji should appear active (either actually active or forced)"""
    return bool(self._force_active or self._is_active)

  @property
  def is_rhd(self) -> bool:
    return self._is_rhd

  def _render(self, _):
    if DEBUG:
      rl.draw_rectangle_lines_ex(self._rect, 1, rl.RED)

    rl.draw_texture(self._dm_background,
                    int(self._rect.x),
                    int(self._rect.y),
                    rl.Color(255, 255, 255, int(255 * self._fade_filter.x)))

    rl.draw_texture(self._dm_person,
                    int(self._rect.x + (self._rect.width - self._dm_person.width) / 2),
                    int(self._rect.y + (self._rect.height - self._dm_person.height) / 2),
                    rl.Color(255, 255, 255, int(255 * 0.9 * self._fade_filter.x)))

    if self.effective_active:
      source_rect = rl.Rectangle(0, 0, self._dm_cone.width, self._dm_cone.height)
      dest_rect = rl.Rectangle(
        self._rect.x + self._rect.width / 2,
        self._rect.y + self._rect.height / 2,
        self._dm_cone.width,
        self._dm_cone.height,
      )

      if not self._lines:
        rl.draw_texture_pro(
          self._dm_cone,
          source_rect,
          dest_rect,
          rl.Vector2(dest_rect.width / 2, dest_rect.height / 2),
          self._rotation_filter.x - 90,
          rl.Color(255, 255, 255, int(255 * self._fade_filter.x * (1 - self._looking_center_filter.x))),
        )

        rl.draw_texture_ex(
          self._dm_center,
          (int(self._rect.x + (self._rect.width - self._dm_center.width) / 2),
           int(self._rect.y + (self._rect.height - self._dm_center.height) / 2)),
          0,
          1.0,
          rl.Color(255, 255, 255, int(255 * self._fade_filter.x * self._looking_center_filter.x)),
        )

      else:
        # remove old angles
        for angle, f in self._head_angles.items():
          dst_from_current = ((angle - self._rotation_filter.x) % 360) - 180
          target = 1.0 if abs(dst_from_current) <= self.LINES_ANGLE_INCREMENT * 5 else 0.0
          if not self._face_detected:
            target = 0.0

          # Reduce all line lengths when looking center
          if self._looking_center:
            target = np.interp(self._looking_center_filter.x, [0.0, 1.0], [target, 0.45])

          f.update(target)
          self._draw_line(angle, f, self._looking_center)

  def _draw_line(self, angle: int, f: FirstOrderFilter, grey: bool):
    line_length = self._rect.width / 6
    line_length = round(np.interp(f.x, [0.0, 1.0], [0, line_length]))
    line_offset = self._rect.width / 2 - line_length * 2  # ensure line ends within rect
    center_x = self._rect.x + self._rect.width / 2
    center_y = self._rect.y + self._rect.height / 2
    start_x = center_x + (line_offset + line_length) * math.cos(math.radians(angle))
    start_y = center_y + (line_offset + line_length) * math.sin(math.radians(angle))
    end_x = start_x + line_length * math.cos(math.radians(angle))
    end_y = start_y + line_length * math.sin(math.radians(angle))
    color = rl.Color(0, 255, 64, 255)

    if grey:
      color = rl.Color(166, 166, 166, 255)

    if f.x > 0.01:
      rl.draw_line_ex((start_x, start_y), (end_x, end_y), 12, color)

  def get_driver_data(self):
    sm = ui_state.sm

    dm_state = sm["driverMonitoringState"]
    self._is_active = dm_state.isActiveMode
    self._is_rhd = dm_state.isRHD
    self._face_detected = dm_state.faceDetected

    driverstate = sm["driverStateV2"]
    driver_data = driverstate.rightDriverData if self._is_rhd else driverstate.leftDriverData
    return driver_data

  def _update_state(self):
    # Get monitoring state
    driver_data = self.get_driver_data()
    driver_orient = driver_data.faceOrientation

    if len(driver_orient) != 3:
      return

    pitch, yaw, roll = driver_orient
    pitch = self._pitch_filter.update(pitch)
    yaw = self._yaw_filter.update(yaw)

    # hysteresis on looking center
    if abs(pitch) < LOOKING_CENTER_THRESHOLD_LOWER and abs(yaw) < LOOKING_CENTER_THRESHOLD_LOWER:
      self._looking_center = True
    elif abs(pitch) > LOOKING_CENTER_THRESHOLD_UPPER or abs(yaw) > LOOKING_CENTER_THRESHOLD_UPPER:
      self._looking_center = False
    self._looking_center_filter.update(1 if self._looking_center else 0)

    if DEBUG:
      pitchd = math.degrees(pitch)
      yawd = math.degrees(yaw)
      rolld = math.degrees(roll)

      rl.draw_line_ex((0, 100), (200, 100), 3, rl.RED)
      rl.draw_line_ex((0, 120), (200, 120), 3, rl.RED)
      rl.draw_line_ex((0, 140), (200, 140), 3, rl.RED)

      pitch_x = 100 + pitchd
      yaw_x = 100 + yawd
      roll_x = 100 + rolld
      rl.draw_circle(int(pitch_x), 100, 5, rl.GREEN)
      rl.draw_circle(int(yaw_x), 120, 5, rl.GREEN)
      rl.draw_circle(int(roll_x), 140, 5, rl.GREEN)

    # filter head rotation, handling wrap-around
    rotation = math.degrees(math.atan2(pitch, yaw))
    angle_diff = rotation - self._rotation_filter.x
    angle_diff = ((angle_diff + 180) % 360) - 180
    self._rotation_filter.update(self._rotation_filter.x + angle_diff)

    if not self.should_draw:
      self._fade_filter.update(0.0)
    elif not self.effective_active:
      self._fade_filter.update(0.35)
    else:
      self._fade_filter.update(1.0)
