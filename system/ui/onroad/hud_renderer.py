import pyray as rl
from dataclasses import dataclass
from cereal.messaging import SubMaster
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.common.conversions import Conversions as CV
from enum import IntEnum

# Constants
SET_SPEED_NA = 255
KM_TO_MILE = 0.621371


@dataclass(frozen=True)
class UIConfig:
  header_height: int = 300
  border_size: int = 30
  button_size: int = 192
  set_speed_width_metric: int = 200
  set_speed_width_imperial: int = 172
  set_speed_height: int = 204
  wheel_icon_size: int = 144


@dataclass(frozen=True)
class FontSizes:
  current_speed: int = 176
  speed_unit: int = 66
  max_speed: int = 40
  set_speed: int = 90


@dataclass(frozen=True)
class Colors:
  white: rl.Color = rl.Color(255, 255, 255, 255)
  disengaged: rl.Color = rl.Color(145, 155, 149, 255)
  override: rl.Color = rl.Color(145, 155, 149, 255)  # Added
  engaged: rl.Color = rl.Color(128, 216, 166, 255)
  disengaged_bg: rl.Color = rl.Color(0, 0, 0, 153)
  override_bg: rl.Color = rl.Color(145, 155, 149, 204)
  engaged_bg: rl.Color = rl.Color(128, 216, 166, 204)
  grey: rl.Color = rl.Color(166, 166, 166, 255)
  dark_grey: rl.Color = rl.Color(114, 114, 114, 255)
  black_translucent: rl.Color = rl.Color(0, 0, 0, 166)
  white_translucent: rl.Color = rl.Color(255, 255, 255, 200)
  border_translucent: rl.Color = rl.Color(255, 255, 255, 75)
  header_gradient_start: rl.Color = rl.Color(0, 0, 0, 114)
  header_gradient_end: rl.Color = rl.Color(0, 0, 0, 0)


UI_CONFIG = UIConfig()
FONT_SIZES = FontSizes()
COLORS = Colors()


class HudStatus(IntEnum):
  DISENGAGED = 0
  OVERRIDE = 1
  ENGAGED = 2


class HudRenderer:
  def __init__(self):
    """Initialize the HUD renderer."""
    self.is_metric: bool = False
    self.status: HudStatus = HudStatus.DISENGAGED
    self.is_cruise_set: bool = False
    self.is_cruise_available: bool = False
    self.set_speed: float = SET_SPEED_NA
    self.speed: float = 0.0
    self.v_ego_cluster_seen: bool = False
    self.font_metrics_cache: dict[[str, int, str], rl.Vector2] = {}
    self._wheel_texture: rl.Texture = gui_app.texture('icons/chffr_wheel.png', UI_CONFIG.wheel_icon_size, UI_CONFIG.wheel_icon_size)
    self._font_semi_bold: rl.Font = gui_app.font(FontWeight.SEMI_BOLD)
    self._font_bold: rl.Font = gui_app.font(FontWeight.BOLD)
    self._font_medium: rl.Font = gui_app.font(FontWeight.MEDIUM)

  def _update_state(self, sm: SubMaster) -> None:
    """Update HUD state based on car state and controls state."""
    self.is_metric = True
    self.status = HudStatus.DISENGAGED

    if not sm.valid['carState']:
      self.is_cruise_set = False
      self.set_speed = SET_SPEED_NA
      self.speed = 0.0
      return

    controls_state = sm['controlsState']
    car_state = sm['carState']

    v_cruise_cluster = car_state.vCruiseCluster
    self.set_speed = (
      controls_state.vCruiseDEPRECATED if v_cruise_cluster == 0.0 else v_cruise_cluster
    )
    self.is_cruise_set = 0 < self.set_speed < SET_SPEED_NA
    self.is_cruise_available = self.set_speed != -1

    if self.is_cruise_set and not self.is_metric:
      self.set_speed *= KM_TO_MILE

    v_ego_cluster = car_state.vEgoCluster
    self.v_ego_cluster_seen = self.v_ego_cluster_seen or v_ego_cluster != 0.0
    v_ego = v_ego_cluster if self.v_ego_cluster_seen else car_state.vEgo
    speed_conversion = CV.MS_TO_KPH if self.is_metric else CV.MS_TO_MPH
    self.speed = max(0.0, v_ego * speed_conversion)

  def draw(self, rect: rl.Rectangle, sm: SubMaster) -> None:
    """Render HUD elements to the screen."""
    self._update_state(sm)
    rl.draw_rectangle_gradient_v(
      int(rect.x),
      int(rect.y),
      int(rect.width),
      UI_CONFIG.header_height,
      COLORS.header_gradient_start,
      COLORS.header_gradient_end,
    )

    if self.is_cruise_available:
      self._draw_set_speed(rect)

    self._draw_current_speed(rect)
    self._draw_wheel_icon(rect)

  def _draw_set_speed(self, rect: rl.Rectangle) -> None:
    """Draw the MAX speed indicator box."""
    set_speed_width = UI_CONFIG.set_speed_width_metric if self.is_metric else UI_CONFIG.set_speed_width_imperial
    x = rect.x + 60 + (UI_CONFIG.set_speed_width_imperial - set_speed_width) // 2
    y = rect.y + 45

    set_speed_rect = rl.Rectangle(x, y, set_speed_width, UI_CONFIG.set_speed_height)
    rl.draw_rectangle_rounded(set_speed_rect, 0.2, 30, COLORS.black_translucent)
    rl.draw_rectangle_rounded_lines_ex(set_speed_rect, 0.2, 30, 6, COLORS.border_translucent)

    max_color = COLORS.grey
    set_speed_color = COLORS.dark_grey
    if self.is_cruise_set:
      set_speed_color = COLORS.white
      max_color = {
        HudStatus.DISENGAGED: COLORS.disengaged,
        HudStatus.OVERRIDE: COLORS.override,
        HudStatus.ENGAGED: COLORS.engaged,
      }.get(self.status, COLORS.grey)

    max_text = "MAX"
    max_text_width = self._measure_text(max_text, self._font_semi_bold, FONT_SIZES.max_speed, 'semi_bold').x
    rl.draw_text_ex(
      self._font_semi_bold,
      max_text,
      rl.Vector2(x + (set_speed_width - max_text_width) / 2, y + 27),
      FONT_SIZES.max_speed,
      0,
      max_color,
    )

    set_speed_text = "–" if not self.is_cruise_set else str(round(self.set_speed))
    speed_text_width = self._measure_text(set_speed_text, self._font_bold, FONT_SIZES.set_speed, 'bold').x
    rl.draw_text_ex(
      self._font_bold,
      set_speed_text,
      rl.Vector2(x + (set_speed_width - speed_text_width) / 2, y + 77),
      FONT_SIZES.set_speed,
      0,
      set_speed_color,
    )

  def _draw_current_speed(self, rect: rl.Rectangle) -> None:
    """Draw the current vehicle speed and unit."""
    speed_text = str(round(self.speed))
    speed_text_size = self._measure_text(speed_text, self._font_bold, FONT_SIZES.current_speed, 'bold')
    speed_pos = rl.Vector2(rect.x + rect.width / 2 - speed_text_size.x / 2, 180 - speed_text_size.y / 2)
    rl.draw_text_ex(self._font_bold, speed_text, speed_pos, FONT_SIZES.current_speed, 0, COLORS.white)

    unit_text = "km/h" if self.is_metric else "mph"
    unit_text_size = self._measure_text(unit_text, self._font_medium, FONT_SIZES.speed_unit, 'medium')
    unit_pos = rl.Vector2(rect.x + rect.width / 2 - unit_text_size.x / 2, 290 - unit_text_size.y / 2)
    rl.draw_text_ex(self._font_medium, unit_text, unit_pos, FONT_SIZES.speed_unit, 0, COLORS.white_translucent)

  def _draw_wheel_icon(self, rect: rl.Rectangle) -> None:
    """Draw the steering wheel icon with status-based opacity."""
    center_x = int(rect.x + rect.width - UI_CONFIG.border_size - UI_CONFIG.button_size / 2)
    center_y = int(rect.y + UI_CONFIG.border_size + UI_CONFIG.button_size / 2)
    rl.draw_circle(center_x, center_y, UI_CONFIG.button_size / 2, COLORS.black_translucent)

    opacity = 0.7 if self.status == HudStatus.DISENGAGED else 1.0
    img_pos = rl.Vector2(center_x - self._wheel_texture.width / 2, center_y - self._wheel_texture.height / 2)
    rl.draw_texture_v(self._wheel_texture, img_pos, rl.Color(255, 255, 255, int(255 * opacity)))

  def _measure_text(self, text: str, font: rl.Font, font_size: int, font_type: str) -> rl.Vector2:
    """Measure text dimensions with caching."""
    key = (text, font_size, font_type)
    if key not in self.font_metrics_cache:
      self.font_metrics_cache[key] = rl.measure_text_ex(font, text, font_size, 0)
    return self.font_metrics_cache[key]
