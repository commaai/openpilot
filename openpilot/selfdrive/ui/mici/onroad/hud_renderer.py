import math
import numpy as np
import raylib_nb as rl
from dataclasses import dataclass
from openpilot.common.constants import CV
from openpilot.selfdrive.ui.mici.onroad import DIMMED_ICON_ALPHA
from openpilot.selfdrive.ui.mici.onroad.status import STATUS_FADE_RC, lerp_color, status_fade
from openpilot.selfdrive.ui.mici.onroad.torque_bar import TorqueBar
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus, ChestnutState
from openpilot.system.ui.lib.application import gui_app, FontWeight, ASSETS_DIR
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from openpilot.common.filter_simple import FirstOrderFilter


# Constants
SET_SPEED_NA = 255
KM_TO_MILE = 0.621371
CRUISE_DISABLED_CHAR = '–'

SET_SPEED_PERSISTENCE = 2.5  # seconds
WHEEL_SIZE = 52
WHEEL_WHITE = rl.Color(255, 255, 255, 255)
WHEEL_GREEN = rl.Color(0, 255, 64, 255)
WHEEL_ORANGE = rl.Color(255, 115, 0, 255)


def _texture(rgba: np.ndarray, px: float) -> rl.Texture:
  texture = rl.load_texture_from_image(rl.Image(None, rgba.shape[1], rgba.shape[0], 1, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8))
  rl.update_texture(texture, np.ascontiguousarray(rgba))
  rl.set_texture_filter(texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
  texture.width, texture.height = round(rgba.shape[1] / px), round(rgba.shape[0] / px)
  return texture


def _wheel_highlight_texture(asset: str, size: int) -> rl.Texture:
  """Center highlight for the wheel, drawn in white over the tinted wheel. Matches the design's Green Glow and Orange Glow styles:
  a radial gradient from the base color mixed 50% with white at the center to the base color at the rim."""
  image = gui_app._load_image_from_path((ASSETS_DIR / asset).as_posix(), size, size)
  w, h, px = image.width, image.height, image.width / size
  alpha = np.frombuffer(image.to_bytes(w * h * 4), dtype=np.uint8).reshape(h, w, 4)[..., 3] / 255.0
  rl.unload_image(image)

  yy, xx = np.mgrid[:h, :w]
  r = np.hypot(xx + 0.5 - w / 2, yy + 0.5 - h / 2) / (w / 2)
  highlight_rgba = np.full((h, w, 4), 255, dtype=np.uint8)
  highlight_rgba[..., 3] = np.clip(alpha * 128 * (1 - r), 0, 255)
  return _texture(highlight_rgba, px)


@dataclass(frozen=True)
class FontSizes:
  current_speed: int = 176
  speed_unit: int = 66
  max_speed: int = 36
  set_speed: int = 112
  shadow_speed: int = 46


@dataclass(frozen=True)
class Colors:
  WHITE = rl.WHITE
  WHITE_TRANSLUCENT = rl.Color(255, 255, 255, 200)


FONT_SIZES = FontSizes()
COLORS = Colors()


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

    self._torque_bar = TorqueBar()

    self._txt_wheel: rl.Texture = gui_app.texture('icons_mici/wheel.png', WHEEL_SIZE, WHEEL_SIZE)
    self._txt_wheel_highlight = _wheel_highlight_texture('icons_mici/wheel.png', WHEEL_SIZE)
    self._txt_chestnut: rl.Texture = gui_app.texture('icons_mici/chestnut.png', 48, 36)
    self._txt_chestnut_green: rl.Texture = gui_app.texture('icons_mici/chestnut_green.png', 48, 36)
    self._txt_chestnut_orange: rl.Texture = gui_app.texture('icons_mici/chestnut_orange.png', 61, 36)
    self._chestnut_icon: rl.Texture | None = None
    self._wheel_critical_filter = FirstOrderFilter(0, STATUS_FADE_RC, 1 / gui_app.target_fps)
    self._wheel_angle_filter = FirstOrderFilter(0, 0.1, 1 / gui_app.target_fps)

    self._set_speed_alpha_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._set_speed_alpha = 0.0
    self._chestnut_alpha_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._txt_shadow_bars = [gui_app.texture(f'icons_mici/longitudinal/distance_{i}.png', w, h, keep_aspect_ratio=False)
                             for i, (w, h) in enumerate(((33, 8), (40, 9), (48, 10)), start=1)]
    self._shadow_alpha_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)

  def set_wheel_critical_icon(self, critical: bool):
    """Set the wheel icon to critical or normal state."""
    self._show_wheel_critical = critical

  def set_torque_bar_hidden(self, hidden: bool):
    self._torque_bar.set_hidden(hidden)

  def set_can_draw_top_icons(self, can_draw_top_icons: bool):
    """Set whether to draw the top part of the HUD."""
    self._can_draw_top_icons = can_draw_top_icons

  def drawing_top_icons(self) -> bool:
    # whether we're drawing any top icons currently
    return bool(self._set_speed_alpha > 1e-2)

  def _update_state(self) -> None:
    """Update HUD state based on car state and controls state."""
    sm = ui_state.sm
    if not ui_state.received_since_started("carState"):
      self.is_cruise_set = False
      self.set_speed = SET_SPEED_NA
      self.speed = 0.0
      return

    car_state = sm['carState']

    set_speed = car_state.vCruiseCluster
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
    self._draw_shadow_speeds(rect)

  def render_steering_wheel(self, rect: rl.Rectangle) -> None:
    """Draw the steering wheel centered at the bottom of rect, after render() has updated state."""
    self._draw_steering_wheel(rect)

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

    pos = rl.Vector2(rect.x + 14, rect.y + rect.height - 10 - icon.height)
    rl.draw_texture_ex(icon, pos, 0.0, 1.0, rl.Color(255, 255, 255, int(255 * opacity * alpha)))

  def _draw_steering_wheel(self, rect: rl.Rectangle) -> None:
    # Green when engaged, orange for steerRequired alerts while engaged, and no color when disengaged.
    critical = self._wheel_critical_filter.update(self._show_wheel_critical and ui_state.status != UIStatus.DISENGAGED)
    status_color = status_fade.color({UIStatus.ENGAGED: WHEEL_GREEN, UIStatus.OVERRIDE: WHEEL_WHITE, UIStatus.DISENGAGED: WHEEL_WHITE})
    wheel_color = lerp_color(status_color, WHEEL_ORANGE, critical)
    engaged = status_fade.weight(UIStatus.ENGAGED)
    highlight = engaged + (1 - engaged) * critical
    wheel_txt = self._txt_wheel

    # pos
    pos_x = int(rect.x + rect.width / 2)
    pos_y = int(rect.y + rect.height - 10 - wheel_txt.height / 2)
    # ease between the live angle and rest on ignition changes; carState keeps its last angle once stopped
    rotation = -ui_state.sm['carState'].steeringAngleDeg * self._wheel_angle_filter.update(ui_state.started and ui_state.received_since_started('carState'))

    alpha = 255 * (1 - (1 - DIMMED_ICON_ALPHA) * status_fade.weight(UIStatus.DISENGAGED) * (1 - critical))
    rl.draw_texture_pro(wheel_txt, rl.Rectangle(0, 0, wheel_txt.width, wheel_txt.height),
                        rl.Rectangle(pos_x, pos_y, wheel_txt.width, wheel_txt.height),
                        rl.Vector2(wheel_txt.width / 2, wheel_txt.height / 2), rotation, rl.Color(wheel_color.r, wheel_color.g, wheel_color.b, int(alpha)))
    rl.draw_texture_pro(self._txt_wheel_highlight, rl.Rectangle(0, 0, wheel_txt.width, wheel_txt.height),
                        rl.Rectangle(pos_x, pos_y, wheel_txt.width, wheel_txt.height),
                        rl.Vector2(wheel_txt.width / 2, wheel_txt.height / 2), rotation, rl.Color(255, 255, 255, int(alpha * highlight)))

  def _draw_shadow_speeds(self, rect: rl.Rectangle) -> None:
    """Draw aggressive/standard/relaxed shadow speeds top-right, clear of the confidence ball and hidden under alerts."""
    plan = ui_state.sm['longitudinalPlan']
    has_speeds = len(plan.shadowSpeeds) == 3
    alpha = self._shadow_alpha_filter.update(ui_state.started and ui_state.has_longitudinal_control and has_speeds and self._can_draw_top_icons)
    if alpha < 1e-2 or not has_speeds:
      return

    conversion = CV.MS_TO_KPH if ui_state.is_metric else CV.MS_TO_MPH
    personality = ui_state.sm['selfdriveState'].personality.raw
    alpha *= 1 - (1 - DIMMED_ICON_ALPHA) * status_fade.weight(UIStatus.DISENGAGED)

    right, bar_center, top, row_h = rect.x + rect.width - 52, rect.x + rect.width - 180, rect.y + 8, 42
    rl.draw_rectangle_rounded(rl.Rectangle(bar_center - 34, top - 4, right - bar_center + 42, 3 * row_h + 8),
                              0.3, 8, rl.Color(0, 0, 0, int(90 * alpha)))
    for i, (speed, bar) in enumerate(zip(plan.shadowSpeeds, self._txt_shadow_bars, strict=True)):
      selected = i == personality
      row_alpha = int(255 * alpha * (0.95 if selected else 0.55))
      slowing = selected and plan.personalitySwitching
      boosting = selected and plan.personalitySwitchBoosting and not slowing
      base = WHEEL_ORANGE if slowing else WHEEL_GREEN if boosting else WHEEL_WHITE
      tint = rl.Color(base.r, base.g, base.b, row_alpha)

      text = str(round(max(speed, 0.0) * conversion))
      size = measure_text_cached(self._font_semi_bold, text, FONT_SIZES.shadow_speed)
      y = top + i * row_h
      rl.draw_text_ex(self._font_semi_bold, text, rl.Vector2(right - size.x, y + (row_h - size.y) / 2), FONT_SIZES.shadow_speed, 0, tint)
      if slowing or boosting:
        aw, ah = 16, 14
        ax, ay = max(right - size.x - 6 - aw, bar_center + 28), y + (row_h - ah) / 2
        verts = [(ax, ay), (ax + aw / 2, ay + ah), (ax + aw, ay)] if slowing else [(ax + aw, ay + ah), (ax + aw / 2, ay), (ax, ay + ah)]
        rl.draw_triangle_fan(verts, 3, tint)
      rl.draw_texture_ex(bar, rl.Vector2(bar_center - bar.width / 2, y + (row_h - bar.height) / 2), 0.0, 1.0, rl.Color(255, 255, 255, row_alpha))

  def _draw_set_speed(self, rect: rl.Rectangle) -> None:
    """Draw the MAX speed indicator box."""
    alpha = self._set_speed_alpha_filter.update(0 < rl.get_time() - self._set_speed_changed_time < SET_SPEED_PERSISTENCE and self._can_draw_top_icons)
    alpha = self._set_speed_alpha = alpha * (1 - status_fade.weight(UIStatus.DISENGAGED))
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
