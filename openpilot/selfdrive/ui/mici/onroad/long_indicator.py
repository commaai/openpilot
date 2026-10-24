import pyray as rl
from openpilot.cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget

HIGHLIGHT_TIME = 2.5  # seconds


class LongIndicator(Widget):
  def __init__(self):
    super().__init__()
    self._txt_lead_car = (self._texture('car', 35, 27), self._texture('car_green', 50, 42))
    self._txt_distance = [(self._texture('distance_1', 32, 7), self._texture('distance_1_green', 60, 35)),
                          (self._texture('distance_2', 40, 9), self._texture('distance_2_green', 68, 37)),
                          (self._texture('distance_3', 48, 11), self._texture('distance_3_green', 76, 39))]
    self._alpha_filter = FirstOrderFilter(0.0, 0.05, 1 / gui_app.target_fps)
    # crossfade between states, a white and a green filter per icon
    self._lead_car_filters = (FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps), FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps))
    self._distance_filters = [(FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps), FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps))
                              for _ in range(3)]
    self._personality: int | None = None
    self._personality_changed_time = -HIGHLIGHT_TIME
    self._should_draw = False

  @staticmethod
  def _texture(name: str, width: int, height: int) -> rl.Texture:
    return gui_app.texture(f'icons_mici/longitudinal/{name}.png', width, height, keep_aspect_ratio=False)

  def set_should_draw(self, should_draw: bool):
    self._should_draw = should_draw

  def _render(self, rect: rl.Rectangle) -> None:
    sm = ui_state.sm
    if sm.recv_frame['selfdriveState'] < ui_state.started_frame or not sm['selfdriveState'].enabled or not ui_state.has_longitudinal_control:
      self._personality = None
      self._personality_changed_time = -HIGHLIGHT_TIME
      self._alpha_filter.x = 0.0
      return

    # hidden under alerts and set speed
    visible = self._should_draw and sm['selfdriveState'].alertSize == log.SelfdriveState.AlertSize.none
    alpha = self._alpha_filter.update(visible)
    self._draw_lead_car(rect, alpha)
    self._draw_distance_bars(rect, alpha, visible)

  def _draw_lead_car(self, rect: rl.Rectangle, alpha: float) -> None:
    sm = ui_state.sm
    plan = sm['longitudinalPlan']
    has_lead = sm.alive['longitudinalPlan'] and plan.hasLead

    e2e = has_lead and plan.longitudinalPlanSource == log.LongitudinalPlan.LongitudinalPlanSource.e2e
    white_f, green_f = self._lead_car_filters
    white_alpha = white_f.update(0.0 if e2e else 0.9 if has_lead else 0.35)
    green_alpha = green_f.update(float(e2e))

    white, green = self._txt_lead_car
    self._draw_centered(white, rect, 100, white_alpha * alpha)
    self._draw_centered(green, rect, 100, green_alpha * alpha)

  def _draw_distance_bars(self, rect: rl.Rectangle, alpha: float, visible: bool) -> None:
    sm = ui_state.sm
    now = rl.get_time()
    personality = sm['selfdriveState'].personality.raw
    if self._personality is not None and personality != self._personality:
      self._personality_changed_time = now
    self._personality = personality
    # the personality alert covers the bars, hold the highlight until they show
    if not visible and now - self._personality_changed_time < HIGHLIGHT_TIME:
      self._personality_changed_time = now
    highlight = now - self._personality_changed_time < HIGHLIGHT_TIME

    # blink at double the turn signal rate (2.67 Hz) while overriding the gas
    overriding = any(e.name == log.OnroadEvent.EventName.gasPressedOverride for e in sm['onroadEvents'])
    blink = 0.35 / 0.9 if overriding and now % 0.375 > 0.1875 else 1.0

    count = personality + 1
    for i, ((white, green), y, (active_f, green_f)) in enumerate(zip(self._txt_distance, (122, 136, 152), self._distance_filters, strict=True)):
      active = active_f.update(float(i < count))
      green_alpha = green_f.update(float(highlight and i == count - 1))
      # only lit bars blink on override
      self._draw_centered(white, rect, y, (0.35 * (1 - active) + 0.9 * (active - green_alpha) * blink) * alpha)
      self._draw_centered(green, rect, y, green_alpha * blink * alpha)

  @staticmethod
  def _draw_centered(texture: rl.Texture, rect: rl.Rectangle, y: float, alpha: float) -> None:
    pos = rl.Vector2(rect.x + 46 - texture.width / 2, rect.y + y - texture.height / 2)
    rl.draw_texture_ex(texture, pos, 0.0, 1.0, rl.Color(255, 255, 255, round(255 * alpha)))
