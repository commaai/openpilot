import math
import raylib_nb as rl
from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.mici.onroad import DIMMED_ICON_ALPHA, alpha_texture, gaussian_blur
from openpilot.selfdrive.ui.mici.onroad.status import status_fade
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

SET_SPEED_NA = 255
SHOW_TIME = 2.5  # seconds to show the set speed and personality after they change
SOURCE_HOLD = 1.0  # seconds the planner source must hold before the icon or speed switches
LongitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource
LEAD_SOURCES = (LongitudinalPlanSource.lead0, LongitudinalPlanSource.lead1, LongitudinalPlanSource.lead2)

# centers in the 72px column. the dial's needle sticks out on the left, so it sits 2px left of center
ICON_Y, ICON_Y_NO_SPEED = 103.5, 120
SPEED_Y = 148.5
SPEED_FONT_SIZE = 36  # Inter Bold 36 in Figma
PERSONALITY_FONT_SIZE, PERSONALITY_SUB_FONT_SIZE = 67, 37  # Inter 64 and 36 in Figma
PERSONALITY_SUBTITLE = "driving personality"
SHADOW_SIGMA, SHADOW_DY = 8 * 10 / 24, 2  # Figma's 0 2 8 black text shadow. Figma blurs 24 as sigma 10
PERSONALITY_NAMES = {
  log.LongitudinalPersonality.aggressive: "aggressive",
  log.LongitudinalPersonality.standard: "standard",
  log.LongitudinalPersonality.relaxed: "relaxed",
}


def _text_shadow(font: rl.Font, text: str, font_size: int, spacing: float) -> tuple[rl.Texture, int]:
  """Blurred black coverage of the text, offset by its padding. Raylib can't draw text off screen,
  so this draws it on the screen before the first frame clears it, and must run outside a frame"""
  size = measure_text_cached(font, text, font_size, spacing)
  rl.clear_background(rl.BLACK)
  rl.draw_text_ex(font, text, rl.Vector2(0, 0), font_size, spacing, rl.WHITE)
  alpha = rl.cpu.screen()[::-1][:math.ceil(size.y), :math.ceil(size.x), 0] / 255.0  # the screen's rows are bottom up
  rl.clear_background(rl.BLACK)
  return alpha_texture(gaussian_blur(alpha, SHADOW_SIGMA), color=(0, 0, 0)), math.ceil(3 * SHADOW_SIGMA)


def _filter(x: float = 0.0, rc: float = 0.1) -> FirstOrderFilter:
  return FirstOrderFilter(x, rc, 1 / gui_app.target_fps)


class LongIndicator(Widget):
  """Speed dial with the set speed below it. openpilot long is green, shows the car while following with the lead policy,
  and shows AUTO when not limited by the set speed."""
  def __init__(self):
    super().__init__()
    self._txt_dial = (self._texture('speed_dial', 54, 46), self._texture('speed_dial_green', 76, 68))
    self._txt_car = (self._texture('car', 50, 39), self._texture('car_green', 72, 61))
    self._txt_auto = self._texture('auto', 58, 20)
    self._font = gui_app.font(FontWeight.BOLD)
    self._font_sub = gui_app.font(FontWeight.ROMAN)
    self._shadows = {name: _text_shadow(self._font, name, PERSONALITY_FONT_SIZE, -2) for name in PERSONALITY_NAMES.values()}
    self._shadows[PERSONALITY_SUBTITLE] = _text_shadow(self._font_sub, PERSONALITY_SUBTITLE, PERSONALITY_SUB_FONT_SIZE, 0)

    self._alpha_filter = _filter(rc=0.05)
    self._lead_filter = _filter()
    self._speed_filter = _filter()
    self._auto_filter = _filter()
    self._personality_filter = _filter()

    self._set_speed = SET_SPEED_NA
    self._shown_set_speed = 0.0  # last valid set speed, held while it fades out
    self._engaged = False
    self._set_speed_changed_time = -SHOW_TIME
    self._source = LongitudinalPlanSource.cruise  # held planner source
    self._source_change_time = 0.0
    self._should_draw = True
    self._personality: int | None = None
    self._personality_changed_time = -SHOW_TIME

  @staticmethod
  def _texture(name: str, width: int, height: int) -> rl.Texture:
    return gui_app.texture(f'icons_mici/longitudinal/{name}.png', width, height, keep_aspect_ratio=False)

  def _update_state(self) -> None:
    sm = ui_state.sm
    now = rl.get_time()
    if not ui_state.started or not ui_state.received_since_started('carState'):
      self._set_speed = SET_SPEED_NA
      self._engaged = False
      self._source = LongitudinalPlanSource.cruise
      self._personality = None
      return

    # show the set speed for a bit when it changes or on engage, even under AUTO
    set_speed = sm['carState'].vCruiseCluster
    engaged = sm['selfdriveState'].enabled
    if engaged and (set_speed != self._set_speed or not self._engaged):
      self._set_speed_changed_time = now
    self._set_speed, self._engaged = set_speed, engaged
    if 0 < set_speed < SET_SPEED_NA:
      self._shown_set_speed = set_speed

    # the planner picks the lowest accel of cruise, lead, and e2e. hold it so close calls don't flicker
    source = sm['longitudinalPlan'].longitudinalPlanSource
    if source == self._source:
      self._source_change_time = now
    elif now - self._source_change_time > SOURCE_HOLD:
      self._source = source

    personality = sm['selfdriveState'].personality.raw
    if ui_state.has_longitudinal_control and self._personality is not None and personality != self._personality:
      self._personality_changed_time = now
    self._personality = personality

  def set_should_draw(self, should_draw: bool):
    self._should_draw = should_draw

  def showing_personality(self) -> bool:
    return self._personality_filter.x > 1e-2 or rl.get_time() - self._personality_changed_time < SHOW_TIME

  def _render(self, rect: rl.Rectangle) -> None:
    # offroad shows the disengaged state
    if ui_state.CP is None or (ui_state.started and not ui_state.received_since_started('selfdriveState')):
      self._alpha_filter.x = 0.0
      return

    disengaged = status_fade.weight(UIStatus.DISENGAGED)
    op_long = ui_state.has_longitudinal_control
    alpha = self._alpha_filter.update(float(self._should_draw))
    # grey while disengaged, white for stock ACC and green for openpilot long while engaged
    green = (1 - disengaged) * op_long
    white_alpha = alpha * (1 - (1 - DIMMED_ICON_ALPHA) * disengaged)

    cruise_set = 0 < self._set_speed < SET_SPEED_NA
    speed = self._speed_filter.update(float(cruise_set)) * (1 - disengaged)
    lead = self._lead_filter.update(float(op_long and self._source in LEAD_SOURCES))
    icon_y = ICON_Y_NO_SPEED + (ICON_Y - ICON_Y_NO_SPEED) * speed

    for (white, green_txt), weight, dx in ((self._txt_dial, 1 - lead, -2), (self._txt_car, lead, 0)):
      self._draw_centered(white, rect, dx, icon_y, white_alpha * weight * (1 - green))
      self._draw_centered(green_txt, rect, dx, icon_y, alpha * weight * green)

    if speed < 1e-2:
      return
    show_auto = op_long and self._source != LongitudinalPlanSource.cruise and rl.get_time() - self._set_speed_changed_time > SHOW_TIME
    auto = self._auto_filter.update(float(show_auto))
    text_y = SPEED_Y + (ICON_Y_NO_SPEED - ICON_Y) * (1 - speed)
    self._draw_centered(self._txt_auto, rect, 0, text_y, 0.9 * alpha * speed * auto)
    self._draw_set_speed(rect, text_y, 0.9 * alpha * speed * (1 - auto))

  def _draw_set_speed(self, rect: rl.Rectangle, y: float, alpha: float) -> None:
    if alpha < 1e-2:
      return
    set_speed = self._shown_set_speed if ui_state.is_metric else self._shown_set_speed * CV.KPH_TO_MPH
    text = str(round(set_speed))
    size = measure_text_cached(self._font, text, SPEED_FONT_SIZE, -1)
    pos = rl.Vector2(round(rect.x + (rect.width - size.x) / 2), round(rect.y + y - size.y / 2))
    rl.draw_text_ex(self._font, text, pos, SPEED_FONT_SIZE, -1, rl.Color(255, 255, 255, round(255 * alpha)))

  def render_personality(self, rect: rl.Rectangle, can_draw: bool) -> None:
    """The personality name in the top left of the camera view for a bit after it changes"""
    visible = can_draw and rl.get_time() - self._personality_changed_time < SHOW_TIME
    alpha = self._personality_filter.update(float(visible))
    if alpha < 1e-2 or self._personality is None:
      return

    name = PERSONALITY_NAMES.get(self._personality, "")
    for font, text, font_size, spacing, y, opacity in ((self._font, name, PERSONALITY_FONT_SIZE, -2, 0, 1.0),
                                                       (self._font_sub, PERSONALITY_SUBTITLE, PERSONALITY_SUB_FONT_SIZE, 0, 77, 0.65)):
      pos = rl.Vector2(rect.x + 21, rect.y + y)
      shadow, pad = self._shadows[text]
      rl.draw_texture_ex(shadow, rl.Vector2(pos.x - pad, pos.y - pad + SHADOW_DY), 0.0, 1.0, rl.Color(255, 255, 255, round(255 * opacity * alpha)))
      rl.draw_text_ex(font, text, pos, font_size, spacing, rl.Color(255, 255, 255, round(255 * 0.9 * opacity * alpha)))

  @staticmethod
  def _draw_centered(texture: rl.Texture, rect: rl.Rectangle, dx: float, y: float, alpha: float) -> None:
    if alpha < 1e-2:
      return
    pos = rl.Vector2(rect.x + (rect.width - texture.width) / 2 + dx, rect.y + y - texture.height / 2)
    rl.draw_texture_ex(texture, pos, 0.0, 1.0, rl.Color(255, 255, 255, round(255 * alpha)))
