import math
import pyray as rl
from dataclasses import dataclass
from openpilot.common.constants import CV
from openpilot.selfdrive.ui.mici.onroad.torque_bar import TorqueBar
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus, ChestnutState
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.cereal import log
from opendbc.car.structs import car

EventName = log.OnroadEvent.EventName

# Constants
SET_SPEED_NA = 255
KM_TO_MILE = 0.621371
CRUISE_DISABLED_CHAR = '–'

SET_SPEED_PERSISTENCE = 2.5  # seconds


@dataclass(frozen=True)
class FontSizes:
  current_speed: int = 176
  speed_unit: int = 66
  max_speed: int = 36
  set_speed: int = 112


@dataclass(frozen=True)
class Colors:
  WHITE = rl.WHITE
  WHITE_TRANSLUCENT = rl.Color(255, 255, 255, 200)


FONT_SIZES = FontSizes()
COLORS = Colors()


class TurnIntent(Widget):
  FADE_IN_ANGLE = 30  # degrees

  def __init__(self):
    super().__init__()
    self._pre = False
    self._turn_intent_direction: int = 0

    self._turn_intent_alpha_filter = FirstOrderFilter(0, 0.05, 1 / gui_app.target_fps)
    self._turn_intent_rotation_filter = FirstOrderFilter(0, 0.1, 1 / gui_app.target_fps)

    self._txt_turn_intent_left: rl.Texture = gui_app.texture('icons_mici/turn_intent_left.png', 50, 20)
    self._txt_turn_intent_right: rl.Texture = gui_app.texture('icons_mici/turn_intent_left.png', 50, 20, flip_x=True)

  def _render(self, _):
    if self._turn_intent_alpha_filter.x > 1e-2:
      turn_intent_texture = self._txt_turn_intent_right if self._turn_intent_direction == 1 else self._txt_turn_intent_left
      src_rect = rl.Rectangle(0, 0, turn_intent_texture.width, turn_intent_texture.height)
      dest_rect = rl.Rectangle(self._rect.x + self._rect.width / 2, self._rect.y + self._rect.height / 2,
                               turn_intent_texture.width, turn_intent_texture.height)

      origin = (turn_intent_texture.width / 2, self._rect.height / 2)
      color = rl.Color(255, 255, 255, int(255 * self._turn_intent_alpha_filter.x))
      rl.draw_texture_pro(turn_intent_texture, src_rect, dest_rect, origin, self._turn_intent_rotation_filter.x, color)

  def _update_state(self) -> None:
    sm = ui_state.sm

    left = any(e.name == EventName.preLaneChangeLeft for e in sm['onroadEvents'])
    right = any(e.name == EventName.preLaneChangeRight for e in sm['onroadEvents'])
    if left or right:
      # pre lane change
      if not self._pre:
        self._turn_intent_rotation_filter.x = self.FADE_IN_ANGLE if left else -self.FADE_IN_ANGLE

      self._pre = True
      self._turn_intent_direction = -1 if left else 1
      self._turn_intent_alpha_filter.update(1)
      self._turn_intent_rotation_filter.update(0)
    elif any(e.name == EventName.laneChange for e in sm['onroadEvents']):
      # fade out and rotate away
      self._pre = False
      self._turn_intent_alpha_filter.update(0)

      if self._turn_intent_direction == 0:
        # unknown. missed pre frame?
        self._turn_intent_rotation_filter.update(0)
      else:
        self._turn_intent_rotation_filter.update(self._turn_intent_direction * self.FADE_IN_ANGLE)
    else:
      # didn't complete lane change, just hide
      self._pre = False
      self._turn_intent_direction = 0
      self._turn_intent_alpha_filter.update(0)
      self._turn_intent_rotation_filter.update(0)


class HudRenderer(Widget):
  def __init__(self):
    super().__init__()
    """Initialize the HUD renderer."""
    self.is_cruise_set: bool = False
    self.is_cruise_available: bool = True
    self.set_speed: float = SET_SPEED_NA
    self._set_speed_changed_time: float = 0
    self.speed: float = 0.0
    self.v_ego_cluster_seen: bool = False
    self._engaged: bool = False
    self._chestnut_fade_time: float = 0

    self._can_draw_top_icons = True
    self._show_wheel_critical = False

    self._font_bold: rl.Font = gui_app.font(FontWeight.BOLD)
    self._font_medium: rl.Font = gui_app.font(FontWeight.MEDIUM)
    self._font_semi_bold: rl.Font = gui_app.font(FontWeight.SEMI_BOLD)
    self._font_display: rl.Font = gui_app.font(FontWeight.DISPLAY)

    self._turn_intent = TurnIntent()
    self._torque_bar = TorqueBar()

    self._txt_lead_car = gui_app.texture('icons_mici/longitudinal/car.png')
    self._txt_lead_car_green = gui_app.texture('icons_mici/longitudinal/car_green.png')
    self._txt_lead_car_orange = gui_app.texture('icons_mici/longitudinal/car_orange.png')
    self._lead_car_white_filter = FirstOrderFilter(0.35, 0.1, 1 / gui_app.target_fps)
    self._lead_car_green_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._lead_car_orange_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)

    # White assets exclude the colored variants' glow padding.
    self._distance_icon_parts = [
      (gui_app.texture(f'icons_mici/longitudinal/{name}.png', width, height, keep_aspect_ratio=False), x, y)
      for name, x, y, width, height in (
        ('distance_1', 26, 119, 32, 7),
        ('distance_2', 22, 132, 40, 9),
        ('distance_3', 18, 147, 48, 11),
      )
    ]
    self._distance_green_parts = [
      (gui_app.texture(f'icons_mici/longitudinal/distance_{index}_green.png', width, height, keep_aspect_ratio=False), x, y)
      for index, x, y, width, height in ((1, 12, 105, 60, 35), (2, 8, 118, 68, 37), (3, 4, 133, 76, 39))
    ]
    self._distance_highlight_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._distance_orange_parts = [
      (gui_app.texture(f'icons_mici/longitudinal/distance_{index}_orange.png', width, height, keep_aspect_ratio=False), x, y)
      for index, x, y, width, height in ((1, 12, 105, 60, 35), (2, 8, 118, 68, 37), (3, 4, 133, 76, 39))
    ]
    self._braking_utilization_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._longitudinal_icon_opacity = 0.0
    self._longitudinal_icon_visible = False
    self._accel_override_alpha = 1.0
    # Match DMoji visibility timing without inheriting its inactive-monitoring dimming.
    self._longitudinal_icon_fade = FirstOrderFilter(0.0, 0.05, 1 / gui_app.target_fps)
    self._reset_distance_highlight()

    self._txt_wheel: rl.Texture = gui_app.texture('icons_mici/wheel.png', 50, 50)
    self._txt_wheel_critical: rl.Texture = gui_app.texture('icons_mici/wheel_critical.png', 50, 50)
    self._txt_exclamation_point: rl.Texture = gui_app.texture('icons_mici/exclamation_point.png', 9, 44)
    self._txt_chestnut: rl.Texture = gui_app.texture('icons_mici/chestnut.png', 60, 44)
    self._txt_chestnut_green: rl.Texture = gui_app.texture('icons_mici/chestnut_green.png', 60, 44)
    self._txt_chestnut_orange: rl.Texture = gui_app.texture('icons_mici/chestnut_orange.png', 75, 44)
    self._chestnut_icon: rl.Texture | None = None
    self._wheel_alpha_filter = FirstOrderFilter(0, 0.05, 1 / gui_app.target_fps)
    self._wheel_y_filter = FirstOrderFilter(0, 0.1, 1 / gui_app.target_fps)

    self._set_speed_alpha_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._chestnut_alpha_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)

  def set_wheel_critical_icon(self, critical: bool):
    """Set the wheel icon to critical or normal state."""
    self._show_wheel_critical = critical

  def set_longitudinal_icon_visible(self, visible: bool) -> None:
    self._longitudinal_icon_visible = visible

  def set_can_draw_top_icons(self, can_draw_top_icons: bool):
    """Set whether to draw the top part of the HUD."""
    self._can_draw_top_icons = can_draw_top_icons

  def drawing_top_icons(self) -> bool:
    # whether we're drawing any top icons currently
    return bool(self._set_speed_alpha_filter.x > 1e-2)

  def _update_state(self) -> None:
    """Update HUD state based on car state and controls state."""
    sm = ui_state.sm
    if sm.recv_frame["carState"] < ui_state.started_frame:
      self.is_cruise_set = False
      self.set_speed = SET_SPEED_NA
      self.speed = 0.0
      return

    controls_state = sm['controlsState']
    car_state = sm['carState']

    v_cruise_cluster = car_state.vCruiseCluster
    set_speed = (
      controls_state.deprecated.vCruise if v_cruise_cluster == 0.0 else v_cruise_cluster
    )
    engaged = sm['selfdriveState'].enabled
    if (set_speed != self.set_speed and engaged) or (engaged and not self._engaged):
      self._set_speed_changed_time = rl.get_time()
    if engaged != self._engaged:
      self._chestnut_fade_time = rl.get_time() if engaged else 0
    self._engaged = engaged
    self.set_speed = set_speed
    self.is_cruise_set = 0 < self.set_speed < SET_SPEED_NA
    self.is_cruise_available = self.set_speed != -1

    v_ego_cluster = car_state.vEgoCluster
    self.v_ego_cluster_seen = self.v_ego_cluster_seen or v_ego_cluster != 0.0
    v_ego = v_ego_cluster if self.v_ego_cluster_seen else car_state.vEgo
    speed_conversion = CV.MS_TO_KPH if ui_state.is_metric else CV.MS_TO_MPH
    self.speed = max(0.0, v_ego * speed_conversion)

  def _render(self, rect: rl.Rectangle) -> None:
    """Render HUD elements to the screen."""

    self._torque_bar.render(rect)

    if self.is_cruise_set:
      self._draw_set_speed(rect)

    self._draw_model_source(rect)

    self._draw_steering_wheel(rect)

    # The combined indicator is only visible while engaged.
    self._longitudinal_icon_opacity = self._longitudinal_icon_fade.update(float(self._longitudinal_icon_visible))
    if ui_state.sm.recv_frame['selfdriveState'] >= ui_state.started_frame and ui_state.sm['selfdriveState'].enabled:
      self._accel_override_alpha = self._acceleration_override_opacity()
      icon_rect = rl.Rectangle(rect.x + 4, rect.y, rect.width, rect.height)
      self._draw_lead_car(icon_rect)
      self._draw_distance_bars(icon_rect)
    else:
      self._reset_distance_highlight()
      self._lead_car_white_filter.x = 0.35
      self._lead_car_green_filter.x = 0.0
      self._lead_car_orange_filter.x = 0.0
      self._braking_utilization_filter.x = 0.0

  def _braking_orange_alpha(self) -> float:
    sm = ui_state.sm
    valid = all(sm.valid[s] and sm.alive[s] and sm.recv_frame[s] >= ui_state.started_frame
                for s in ('controlsState', 'carControl'))
    utilization = sm['controlsState'].brakingUtilization if valid and sm['carControl'].longActive else 0.0
    utilization = max(0.0, min(1.0, utilization)) if math.isfinite(utilization) else 0.0
    # Match TorqueBar: filter utilization, then blend from 75% to 100%.
    return max(0.0, self._braking_utilization_filter.update(utilization) - 0.75) * 4.0

  def _acceleration_override_opacity(self) -> float:
    sm = ui_state.sm
    overriding = (sm.valid['onroadEvents'] and sm.alive['onroadEvents'] and
                  sm.recv_frame['onroadEvents'] >= ui_state.started_frame and
                  any(event.name == EventName.gasPressedOverride for event in sm['onroadEvents']))
    # Match the torque bar's steady 35% foreground during override, while
    # retaining the longitudinal indicator's independent policy/warning colors.
    return 0.35 if overriding else 1.0

  def _reset_distance_highlight(self) -> None:
    self._distance_personality = None
    self._personality_highlight_time = -math.inf
    self._distance_highlight_filter.x = 0.0

  def _distance_highlight_alpha(self, personality: int, now: float) -> float:
    if self._distance_personality is not None and personality != self._distance_personality:
      self._personality_highlight_time = now
    self._distance_personality = personality
    # Match the set-speed HUD's persistence and fade for personality changes.
    highlighted = now - self._personality_highlight_time < SET_SPEED_PERSISTENCE
    return self._distance_highlight_filter.update(float(highlighted))

  @staticmethod
  def _longitudinal_layout(personality):
    # Figma positions before the shared 4 px rightward offset. Smaller gaps
    # replace the upper bars with a larger car, leaving the wider lower bars.
    if personality == log.LongitudinalPersonality.aggressive:
      return (18, 95, 48, 38), 2, -8
    if personality == log.LongitudinalPersonality.relaxed:
      return (25, 86, 34, 27), 0, 0
    return (21, 89, 42, 34), 1, -3

  def _draw_distance_bars(self, rect: rl.Rectangle) -> None:
    sm = ui_state.sm
    personality = sm['selfdriveState'].personality
    _, first_bar, y_offset = self._longitudinal_layout(personality)
    green_alpha = self._distance_highlight_alpha(personality.raw, rl.get_time())
    orange_alpha = self._braking_orange_alpha()
    for index, (texture, x, y) in enumerate(self._distance_icon_parts):
      if index < first_bar:
        continue
      highlighted = index == 2
      alpha = 0.9 * (1.0 - green_alpha if highlighted else 1.0)
      alpha *= self._accel_override_alpha * (1.0 - orange_alpha)
      color = rl.Color(255, 255, 255, round(255 * alpha * self._longitudinal_icon_opacity))
      rl.draw_texture_ex(texture, rl.Vector2(rect.x + x, rect.y + y + y_offset), 0.0, 1.0, color)
      if highlighted and green_alpha > 0:
        green, gx, gy = self._distance_green_parts[index]
        rl.draw_texture_ex(green, rl.Vector2(rect.x + gx, rect.y + gy + y_offset), 0.0, 1.0,
                           rl.Color(255, 255, 255, round(255 * green_alpha * (1.0 - orange_alpha) *
                                                        self._longitudinal_icon_opacity * self._accel_override_alpha)))
      if orange_alpha > 0:
        orange, ox, oy = self._distance_orange_parts[index]
        rl.draw_texture_ex(orange, rl.Vector2(rect.x + ox, rect.y + oy + y_offset), 0.0, 1.0,
                           rl.Color(255, 255, 255, round(255 * orange_alpha * self._longitudinal_icon_opacity * self._accel_override_alpha)))

  def _draw_lead_car(self, rect: rl.Rectangle) -> None:
    sm = ui_state.sm
    plan = sm['longitudinalPlan']
    has_lead = (sm.valid['longitudinalPlan'] and sm.alive['longitudinalPlan'] and
                sm.recv_frame['longitudinalPlan'] >= ui_state.started_frame and plan.hasLead)
    fcw = (sm.valid['selfdriveState'] and sm.alive['selfdriveState'] and
           sm.recv_frame['selfdriveState'] >= ui_state.started_frame and
           sm['selfdriveState'].alertHudVisual == car.CarControl.HUDControl.VisualAlert.fcw)
    green = not fcw and has_lead and plan.longitudinalPlanSource == log.LongitudinalPlan.LongitudinalPlanSource.e2e
    white_alpha = self._lead_car_white_filter.update(0.0 if green or fcw else (0.9 if has_lead else 0.35))
    green_alpha = self._lead_car_green_filter.update(float(green))
    orange_alpha = self._lead_car_orange_filter.update(float(fcw))
    override_alpha = self._accel_override_alpha if has_lead else 1.0
    (x, y, width, height), _, _ = self._longitudinal_layout(sm['selfdriveState'].personality)
    # The new 128x101 car has 28 source pixels of glow on every side in
    # the 184x157 colored exports. Scale that padding with the car so color
    # crossfades never change its apparent silhouette or placement.
    pad_x, pad_y = 28 * width / 128, 28 * height / 101
    white_rect = rl.Rectangle(rect.x + x, rect.y + y, width, height)
    glow_rect = rl.Rectangle(rect.x + x - pad_x, rect.y + y - pad_y, width + 2 * pad_x, height + 2 * pad_y)
    for texture, destination, alpha in ((self._txt_lead_car, white_rect, white_alpha),
                                         (self._txt_lead_car_green, glow_rect, green_alpha),
                                         (self._txt_lead_car_orange, glow_rect, orange_alpha)):
      color = rl.Color(255, 255, 255, round(255 * alpha * self._longitudinal_icon_opacity * override_alpha))
      source = rl.Rectangle(0, 0, texture.width, texture.height)
      rl.draw_texture_pro(texture, source, destination, rl.Vector2(0, 0), 0.0, color)

  def _draw_model_source(self, rect: rl.Rectangle) -> None:
    if ui_state.sm.recv_frame['selfdriveState'] < ui_state.started_frame:
      return

    loading = ui_state.chestnut_state == ChestnutState.LOADING
    if loading:
      icon = self._txt_chestnut
      opacity = 0.35 + 0.65 * (0.5 - 0.5 * math.cos(rl.get_time() * 6.0))
    elif ui_state.chestnut_state in (ChestnutState.UNCOMPILED, ChestnutState.FAILED):
      icon = self._txt_chestnut_orange
      opacity = 1.0
    elif ui_state.chestnut_state == ChestnutState.ACTIVE:
      icon = self._txt_chestnut_green
      opacity = 1.0
    else:
      return

    if icon is not self._chestnut_icon:
      self._chestnut_fade_time = rl.get_time()
      self._chestnut_icon = icon
    visible = loading or rl.get_time() - self._chestnut_fade_time < SET_SPEED_PERSISTENCE
    alpha = self._chestnut_alpha_filter.update(visible)
    if alpha < 1e-2:
      return

    pos = rl.Vector2(rect.x + rect.width - 10 - icon.width,
                     rect.y + rect.height - 14 - (self._txt_wheel.height + icon.height) / 2)
    rl.draw_texture_ex(icon, pos, 0.0, 1.0, rl.Color(255, 255, 255, int(255 * opacity * alpha)))

  def _draw_steering_wheel(self, rect: rl.Rectangle) -> None:
    wheel_txt = self._txt_wheel_critical if self._show_wheel_critical else self._txt_wheel

    if self._show_wheel_critical:
      self._wheel_alpha_filter.update(255)
      self._wheel_y_filter.update(0)
    else:
      if ui_state.status == UIStatus.DISENGAGED:
        self._wheel_alpha_filter.update(0)
        self._wheel_y_filter.update(wheel_txt.height / 2)
      else:
        self._wheel_alpha_filter.update(255 * 0.9)
        self._wheel_y_filter.update(0)

    # pos
    pos_x = int(rect.x + 21 + wheel_txt.width / 2)
    pos_y = int(rect.y + rect.height - 14 - wheel_txt.height / 2 + self._wheel_y_filter.x)
    rotation = -ui_state.sm['carState'].steeringAngleDeg

    turn_intent_margin = 25
    self._turn_intent.render(rl.Rectangle(
      pos_x - wheel_txt.width / 2 - turn_intent_margin,
      pos_y - wheel_txt.height / 2 - turn_intent_margin,
      wheel_txt.width + turn_intent_margin * 2,
      wheel_txt.height + turn_intent_margin * 2,
    ))

    src_rect = rl.Rectangle(0, 0, wheel_txt.width, wheel_txt.height)
    dest_rect = rl.Rectangle(pos_x, pos_y, wheel_txt.width, wheel_txt.height)
    origin = (wheel_txt.width / 2, wheel_txt.height / 2)

    # color and draw
    color = rl.Color(255, 255, 255, int(self._wheel_alpha_filter.x))
    rl.draw_texture_pro(wheel_txt, src_rect, dest_rect, origin, rotation, color)

    if self._show_wheel_critical:
      # Draw exclamation point icon
      EXCLAMATION_POINT_SPACING = 10
      exclamation_pos_x = pos_x - self._txt_exclamation_point.width / 2 + wheel_txt.width / 2 + EXCLAMATION_POINT_SPACING
      exclamation_pos_y = pos_y - self._txt_exclamation_point.height / 2
      rl.draw_texture_ex(self._txt_exclamation_point, rl.Vector2(exclamation_pos_x, exclamation_pos_y), 0.0, 1.0, rl.WHITE)

  def _draw_set_speed(self, rect: rl.Rectangle) -> None:
    """Draw the MAX speed indicator box."""
    alpha = self._set_speed_alpha_filter.update(0 < rl.get_time() - self._set_speed_changed_time < SET_SPEED_PERSISTENCE and
                                                self._can_draw_top_icons and self._engaged)
    if alpha < 1e-2:
      return

    x = rect.x
    y = rect.y

    # draw drop shadow
    circle_radius = 162 // 2
    rl.draw_circle_gradient(rl.Vector2(x + circle_radius, y + circle_radius), circle_radius,
                            rl.Color(0, 0, 0, int(255 / 2 * alpha)), rl.BLANK)

    set_speed_color = rl.Color(255, 255, 255, int(255 * 0.9 * alpha))
    max_color = rl.Color(255, 255, 255, int(255 * 0.9 * alpha))

    set_speed = self.set_speed
    if self.is_cruise_set and not ui_state.is_metric:
      set_speed *= KM_TO_MILE

    set_speed_text = CRUISE_DISABLED_CHAR if not self.is_cruise_set else str(round(set_speed))
    rl.draw_text_ex(
      self._font_display,
      set_speed_text,
      rl.Vector2(x + 13 + 4, y + 3 - 8 - 3 + 4),
      FONT_SIZES.set_speed,
      0,
      set_speed_color,
    )

    max_text = tr("MAX")
    rl.draw_text_ex(
      self._font_semi_bold,
      max_text,
      rl.Vector2(x + 25, y + FONT_SIZES.set_speed - 7 + 4),
      FONT_SIZES.max_speed,
      0,
      max_color,
    )

  def _draw_current_speed(self, rect: rl.Rectangle) -> None:
    """Draw the current vehicle speed and unit."""
    speed_text = str(round(self.speed))
    speed_text_size = measure_text_cached(self._font_bold, speed_text, FONT_SIZES.current_speed)
    speed_pos = rl.Vector2(rect.x + rect.width / 2 - speed_text_size.x / 2, 180 - speed_text_size.y / 2)
    rl.draw_text_ex(self._font_bold, speed_text, speed_pos, FONT_SIZES.current_speed, 0, COLORS.WHITE)

    unit_text = tr("km/h") if ui_state.is_metric else tr("mph")
    unit_text_size = measure_text_cached(self._font_medium, unit_text, FONT_SIZES.speed_unit)
    unit_pos = rl.Vector2(rect.x + rect.width / 2 - unit_text_size.x / 2, 290 - unit_text_size.y / 2)
    rl.draw_text_ex(self._font_medium, unit_text, unit_pos, FONT_SIZES.speed_unit, 0, COLORS.WHITE_TRANSLUCENT)
