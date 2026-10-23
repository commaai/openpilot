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

EventName = log.OnroadEvent.EventName

# Constants
SET_SPEED_NA = 255
KM_TO_MILE = 0.621371
CRUISE_DISABLED_CHAR = '–'

SET_SPEED_PERSISTENCE = 2.5  # seconds

WHEEL_SIZE = 52
WHEEL_GLOW_SIZE = round(WHEEL_SIZE * 184 / 128)  # glow assets pad the 128px wheel to 184px


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
  WHEEL_SIZE = 96
  ARROW_WIDTH, ARROW_HEIGHT = 85, 34
  ARROW_OFFSET = 75  # arrow center above the wheel center

  def __init__(self):
    super().__init__()
    self._pre = False
    self._turn_intent_direction: int = 0

    self._turn_intent_alpha_filter = FirstOrderFilter(0, 0.05, 1 / gui_app.target_fps)
    self._turn_intent_rotation_filter = FirstOrderFilter(0, 0.1, 1 / gui_app.target_fps)

    self._txt_turn_intent_left: rl.Texture = gui_app.texture('icons_mici/turn_intent_left.png', self.ARROW_WIDTH, self.ARROW_HEIGHT)
    self._txt_turn_intent_right: rl.Texture = gui_app.texture('icons_mici/turn_intent_left.png', self.ARROW_WIDTH, self.ARROW_HEIGHT, flip_x=True)
    self._txt_wheel: rl.Texture = gui_app.texture('icons_mici/wheel.png', self.WHEEL_SIZE, self.WHEEL_SIZE)

  def _render(self, _):
    if self._turn_intent_alpha_filter.x > 1e-2:
      color = rl.Color(255, 255, 255, int(255 * self._turn_intent_alpha_filter.x))
      turn_intent_texture = self._txt_turn_intent_right if self._turn_intent_direction == 1 else self._txt_turn_intent_left
      # arrow orbits the wheel at the top of the rect
      self._draw_rotated(turn_intent_texture, self._rect.height / 2, self._turn_intent_rotation_filter.x, color)
      self._draw_rotated(self._txt_wheel, self._txt_wheel.height / 2, -ui_state.sm['carState'].steeringAngleDeg, color)

  def _draw_rotated(self, texture: rl.Texture, origin_y: float, rotation: float, color: rl.Color) -> None:
    src_rect = rl.Rectangle(0, 0, texture.width, texture.height)
    dest_rect = rl.Rectangle(self._rect.x + self._rect.width / 2, self._rect.y + self._rect.height / 2, texture.width, texture.height)
    rl.draw_texture_pro(texture, src_rect, dest_rect, (texture.width / 2, origin_y), rotation, color)

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
    self._show_wheel_alert = False

    self._font_bold: rl.Font = gui_app.font(FontWeight.BOLD)
    self._font_medium: rl.Font = gui_app.font(FontWeight.MEDIUM)
    self._font_semi_bold: rl.Font = gui_app.font(FontWeight.SEMI_BOLD)
    self._font_display: rl.Font = gui_app.font(FontWeight.DISPLAY)

    self._turn_intent = TurnIntent()
    self._torque_bar = TorqueBar()

    self._txt_wheel: rl.Texture = gui_app.texture('icons_mici/wheel.png', WHEEL_SIZE, WHEEL_SIZE)
    self._txt_wheel_green: rl.Texture = gui_app.texture('icons_mici/wheel_green.png', WHEEL_GLOW_SIZE, WHEEL_GLOW_SIZE)
    self._txt_wheel_orange: rl.Texture = gui_app.texture('icons_mici/wheel_orange.png', WHEEL_GLOW_SIZE, WHEEL_GLOW_SIZE)
    self._txt_chestnut: rl.Texture = gui_app.texture('icons_mici/chestnut.png', 60, 44)
    self._txt_chestnut_green: rl.Texture = gui_app.texture('icons_mici/chestnut_green.png', 60, 44)
    self._txt_chestnut_orange: rl.Texture = gui_app.texture('icons_mici/chestnut_orange.png', 75, 44)
    self._chestnut_icon: rl.Texture | None = None
    self._wheel_alpha_filter = FirstOrderFilter(0, 0.05, 1 / gui_app.target_fps)

    self._set_speed_alpha_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._chestnut_alpha_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)

  def set_wheel_alert(self, alert: bool):
    self._show_wheel_alert = alert

  def set_can_draw_top_icons(self, can_draw_top_icons: bool):
    """Set whether to draw the top part of the HUD."""
    self._can_draw_top_icons = can_draw_top_icons

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

    # steering hint in the bottom left of the video
    hint_x = rect.x + 16 + TurnIntent.WHEEL_SIZE / 2
    hint_y = rect.y + rect.height - 7 - TurnIntent.WHEEL_SIZE / 2
    hint_radius = TurnIntent.ARROW_OFFSET + TurnIntent.ARROW_HEIGHT / 2
    self._turn_intent.render(rl.Rectangle(hint_x - hint_radius, hint_y - hint_radius, hint_radius * 2, hint_radius * 2))

  def _draw_model_source(self, rect: rl.Rectangle) -> None:
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

    pos = rl.Vector2(rect.x + 10, rect.y + rect.height - 10 - icon.height)
    rl.draw_texture_ex(icon, pos, 0.0, 1.0, rl.Color(255, 255, 255, int(255 * opacity * alpha)))

  def draw_steering_wheel(self, rect: rl.Rectangle) -> None:
    # TODO: be more selective, e.g. DM alerts shouldn't turn the wheel orange while still engaged
    if self._show_wheel_alert:
      wheel_txt, alpha = self._txt_wheel_orange, 1.0
    elif ui_state.status == UIStatus.ENGAGED:
      wheel_txt, alpha = self._txt_wheel_green, 1.0
    elif ui_state.status == UIStatus.OVERRIDE:
      wheel_txt, alpha = self._txt_wheel, 0.9
    else:
      wheel_txt, alpha = self._txt_wheel, 0.35
    self._wheel_alpha_filter.update(255 * alpha)

    # centered at the bottom of the rect
    pos_x = int(rect.x + rect.width / 2)
    pos_y = int(rect.y + rect.height - 10 - WHEEL_SIZE / 2)
    rotation = -ui_state.sm['carState'].steeringAngleDeg

    src_rect = rl.Rectangle(0, 0, wheel_txt.width, wheel_txt.height)
    dest_rect = rl.Rectangle(pos_x, pos_y, wheel_txt.width, wheel_txt.height)
    origin = (wheel_txt.width / 2, wheel_txt.height / 2)

    # color and draw
    color = rl.Color(255, 255, 255, int(self._wheel_alpha_filter.x))
    rl.draw_texture_pro(wheel_txt, src_rect, dest_rect, origin, rotation, color)

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
