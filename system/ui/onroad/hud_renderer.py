import pyray as rl
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.common.conversions import Conversions as cv


# Constants
SET_SPEED_NA = 255
KM_TO_MILE = 0.621371
UI_HEADER_HEIGHT = 300
UI_BORDER_SIZE = 30
WHEEL_ICON_SIZE = 192

# Global Color Definitions
COLOR_WHITE = rl.WHITE
COLOR_DISENGAGED = rl.WHITE
COLOR_OVERRIDE = rl.Color(145, 155, 149, 255)
COLOR_ENGAGED = rl.Color(128, 216, 166, 255)
COLOR_DISENGAGED_BG = rl.Color(0, 0, 0, 153)
COLOR_OVERRIDE_BG = rl.Color(145, 155, 149, 204)
COLOR_ENGAGED_BG = rl.Color(128, 216, 166, 204)
COLOR_GREY = rl.Color(166, 166, 166, 255)
COLOR_DARK_GREY = rl.Color(114, 114, 114, 255)
COLOR_BLACK_TRANSLUCENT = rl.Color(0, 0, 0, 166)
COLOR_WHITE_TRANSLUCENT = rl.Color(255, 255, 255, 200)
COLOR_BORDER_TRANSLUCENT = rl.Color(255, 255, 255, 75)
COLOR_HEADER_GRADIENT_START = rl.Color(0, 0, 0, 114)
COLOR_HEADER_GRADIENT_END = rl.Color(0, 0, 0, 0)


CURRENT_SPEED_FONT_SIZE = 176
SPEED_UNIT_FONT_SIZE = 66
MAX_SPEED_FONT_SIZE = 40
SET_SPEED_FONT_SIZE = 90


class HudRenderer:
  def __init__(self):
    self.is_metric = False
    self.status = 0  # Default status
    self.is_cruise_set = False
    self.is_cruise_available = False
    self.set_speed = SET_SPEED_NA
    self.speed = 0.0
    self.v_ego_cluster_seen = False
    self.font_metrics_cache = {}
    self._wheel_texture = gui_app.texture("icons/chffr_wheel.png", WHEEL_ICON_SIZE, WHEEL_ICON_SIZE)

    self._font_semi_bold = gui_app.font(FontWeight.SEMI_BOLD)
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_medium = gui_app.font(FontWeight.MEDIUM)

  def _update_state(self, sm):
    """Update HUD state based on car state and controls state"""
    self.is_metric = True  # scene.is_metric
    self.status = 1  # scene.status

    if not sm.valid["carState"]:
      self.is_cruise_set = False
      self.set_speed = SET_SPEED_NA
      self.speed = 0.0
      return

    controls_state = sm["controlsState"]
    car_state = sm["carState"]

    # Handle older routes where vCruiseCluster is not set
    v_cruise_cluster = car_state.vCruiseCluster
    self.set_speed = controls_state.vCruiseDEPRECATED if v_cruise_cluster == 0.0 else v_cruise_cluster

    self.is_cruise_set = self.set_speed > 0 and self.set_speed != SET_SPEED_NA
    self.is_cruise_available = self.set_speed != -1

    if self.is_cruise_set and not self.is_metric:
      self.set_speed *= KM_TO_MILE

    # Handle older routes where vEgoCluster is not set
    v_ego_cluster = car_state.vEgoCluster
    self.v_ego_cluster_seen = self.v_ego_cluster_seen or v_ego_cluster != 0.0
    v_ego = v_ego_cluster if self.v_ego_cluster_seen else car_state.vEgo

    # Convert speed to appropriate units
    speed_conversion = cv.MS_TO_KPH if self.is_metric else cv.MS_TO_MPH
    self.speed = max(0.0, v_ego * speed_conversion)

  def draw(self, rect, sm):
    """Draw the HUD elements to the screen"""
    self._update_state(sm)

    # Draw header gradient
    rl.draw_rectangle_gradient_v(
      int(rect.x),
      int(rect.y),
      int(rect.width),
      UI_HEADER_HEIGHT,
      COLOR_HEADER_GRADIENT_START,
      COLOR_HEADER_GRADIENT_END,
    )

    if self.is_cruise_available:
      self._draw_set_speed(rect)

    self._draw_current_speed(rect)
    self._draw_wheel_icon(rect)

  def _draw_set_speed(self, rect):
    """Draw the MAX speed indicator box"""
    default_width = 172
    set_speed_width = 200 if self.is_metric else default_width
    set_speed_height = 204

    x = rect.x + 60 + (default_width - set_speed_width) // 2
    y = rect.y + 45

    # Draw background
    rl.draw_rectangle_rounded(
      rl.Rectangle(x, y, set_speed_width, set_speed_height),
      0.2,  # roundness
      30,  # segments
      COLOR_BLACK_TRANSLUCENT,
    )

    # Draw border
    rl.draw_rectangle_rounded_lines_ex(
      rl.Rectangle(x, y, set_speed_width, set_speed_height),
      0.2,  # roundness
      30,  # segments
      6,  # thickness
      COLOR_BORDER_TRANSLUCENT,
    )

    # Determine text colors based on state
    max_color = COLOR_GREY
    set_speed_color = COLOR_DARK_GREY

    if self.is_cruise_set:
      set_speed_color = COLOR_WHITE
      if self.status == 0:
        max_color = COLOR_DISENGAGED
      elif self.status == 1:
        max_color = COLOR_OVERRIDE
      else:
        max_color = COLOR_ENGAGED

    # Draw "MAX" text
    max_text = "MAX"
    max_text_width = self._measure_text(max_text, self._font_semi_bold, MAX_SPEED_FONT_SIZE).x
    max_x = x + (set_speed_width - max_text_width) / 2
    rl.draw_text_ex(self._font_semi_bold, max_text, rl.Vector2(max_x, y + 27), MAX_SPEED_FONT_SIZE, 0, max_color)

    # Draw set speed value
    set_speed_text = "–" if not self.is_cruise_set else str(round(self.set_speed))
    speed_text_width = self._measure_text(set_speed_text, self._font_bold, SET_SPEED_FONT_SIZE).x
    speed_x = x + (set_speed_width - speed_text_width) / 2
    rl.draw_text_ex(
      self._font_bold, set_speed_text, rl.Vector2(speed_x, y + 77), SET_SPEED_FONT_SIZE, 0, set_speed_color
    )

  def _draw_current_speed(self, rect):
    """Draw the current vehicle speed"""
    speed_text = str(round(self.speed))
    speed_text_size = self._measure_text(speed_text, self._font_bold, CURRENT_SPEED_FONT_SIZE)

    # Position speed text in center of rect
    speed_pos = rl.Vector2(rect.x + rect.width / 2 - speed_text_size.x / 2, 180 - speed_text_size.y / 2)
    rl.draw_text_ex(self._font_bold, speed_text, speed_pos, CURRENT_SPEED_FONT_SIZE, 0, COLOR_WHITE)

    # Draw speed unit
    unit_text = "km/h" if self.is_metric else "mph"
    unit_text_size = self._measure_text(unit_text, self._font_medium, SPEED_UNIT_FONT_SIZE)

    # Position unit text in center of rect
    unit_pos = rl.Vector2(rect.x + rect.width / 2 - unit_text_size.x / 2, 290 - unit_text_size.y / 2)
    rl.draw_text_ex(self._font_medium, unit_text, unit_pos, SPEED_UNIT_FONT_SIZE, 0, COLOR_WHITE_TRANSLUCENT)

  def _draw_wheel_icon(self, rect):
    """Draw the steering wheel icon with status-based background color"""
    center_x = int(rect.x + rect.width - UI_BORDER_SIZE - WHEEL_ICON_SIZE / 2)
    center_y = int(rect.y + UI_BORDER_SIZE + WHEEL_ICON_SIZE / 2)

    bg_colors = [COLOR_DISENGAGED_BG, COLOR_OVERRIDE_BG, COLOR_ENGAGED_BG]
    bg_color = bg_colors[min(self.status, 2)]

    rl.draw_circle(center_x, center_y, WHEEL_ICON_SIZE / 2, bg_color)

    opacity = 0.7 if self.status == 0 else 1.0
    img_pos = rl.Vector2(center_x - self._wheel_texture.width / 2, center_y - self._wheel_texture.height / 2)
    rl.draw_texture_v(self._wheel_texture, img_pos, rl.Color(255, 255, 255, int(255 * opacity)))

  def _measure_text(self, text: str, font: rl.Font, font_size: int) -> rl.Vector2:
    """Get text metrics for a given text and font weight with caching"""
    key = (text, font_size)
    if key not in self.font_metrics_cache:
      metrics = rl.measure_text_ex(font, text, font_size, 0)
      self.font_metrics_cache[key] = metrics

    return self.font_metrics_cache[key]
