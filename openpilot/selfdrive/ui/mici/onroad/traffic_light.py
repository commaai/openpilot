from enum import IntEnum
from collections.abc import Sequence
import raylib_nb as rl
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget

# limits on modelV2.trafficUnits, which is [u69, u248, u201, u239]
HOLD_MIN_SUM = 0.3
BRAKING_MIN_SUM = 1.0
GREEN_MIN = 0.2
GREEN_MAX_OTHERS_RATIO = 0.5  # the other units together must stay below this part of u239

BODY_WIDTH = 34
BODY_HEIGHT = 88
LAMP_RADIUS = 11
MARGIN = 20
BRAKING_BLINK_HZ = 2.0

RED = rl.Color(255, 0, 21, 255)
GREEN = rl.Color(0, 255, 38, 255)
LAMP_OFF = rl.Color(60, 60, 60, 255)
BODY = rl.Color(20, 20, 20, 230)


class TrafficLightState(IntEnum):
  NONE = 0
  GREEN = 1
  BRAKING = 2
  HOLD = 3


def classify_traffic_units(units: Sequence[float]) -> TrafficLightState:
  # an empty list means the model does not have the units
  if len(units) != 4:
    return TrafficLightState.NONE

  u69, u248, u201, u239 = units
  hold = u69 + u248
  total = hold + u201 + u239
  if hold >= max(u201, u239) and total > HOLD_MIN_SUM:
    return TrafficLightState.HOLD
  if u201 >= max(hold, u239) and total > BRAKING_MIN_SUM:
    return TrafficLightState.BRAKING
  if u239 > GREEN_MIN and hold + u201 < GREEN_MAX_OTHERS_RATIO * u239:
    return TrafficLightState.GREEN
  return TrafficLightState.NONE


def _with_alpha(color: rl.Color, alpha: float) -> rl.Color:
  return rl.Color(color.r, color.g, color.b, round(color.a * alpha))


class TrafficLight(Widget):
  def __init__(self):
    super().__init__()
    self._alpha_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._state = TrafficLightState.NONE
    self._should_draw = False

  def set_should_draw(self, should_draw: bool):
    self._should_draw = should_draw

  def _update_state(self):
    super()._update_state()
    state = classify_traffic_units(ui_state.sm['modelV2'].trafficUnits)
    # keep the last color while the icon fades out
    if state != TrafficLightState.NONE:
      self._state = state
    self._alpha_filter.update(float(self._should_draw and state != TrafficLightState.NONE))

  def _render(self, rect: rl.Rectangle):
    alpha = self._alpha_filter.x
    if alpha < 1e-2:
      return

    body = rl.Rectangle(rect.x + rect.width - MARGIN - BODY_WIDTH, rect.y + MARGIN, BODY_WIDTH, BODY_HEIGHT)
    rl.draw_rectangle_rounded(body, 0.5, 10, _with_alpha(BODY, alpha))

    # red blinks while the car brakes for the light, and stays on while the car is held
    red_alpha = 0.0
    if self._state == TrafficLightState.HOLD:
      red_alpha = 1.0
    elif self._state == TrafficLightState.BRAKING:
      blink_phase = rl.get_time() * BRAKING_BLINK_HZ % 1.0
      red_alpha = 0.3 + 0.7 * abs(2.0 * blink_phase - 1.0)
    green_alpha = float(self._state == TrafficLightState.GREEN)

    # top to bottom: red, yellow (never on), green
    center_x = int(body.x + BODY_WIDTH / 2)
    lamp_spacing = BODY_HEIGHT / 3
    for i, (lit_color, lit_alpha) in enumerate(((RED, red_alpha), (LAMP_OFF, 0.0), (GREEN, green_alpha))):
      center_y = int(body.y + lamp_spacing * (i + 0.5))
      rl.draw_circle(center_x, center_y, LAMP_RADIUS, _with_alpha(LAMP_OFF, alpha))
      if lit_alpha > 0.0:
        rl.draw_circle(center_x, center_y, LAMP_RADIUS, _with_alpha(lit_color, alpha * lit_alpha))
