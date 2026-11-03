from enum import IntEnum
from collections.abc import Sequence
import raylib_nb as rl
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget

# limits on modelV2.trafficUnits, which is [u69, u248, u201, u239], smoothed.
# u69/u248 are on whenever the car stands still, at a red light or not. So red while stopped only shows
# if the car braked for a light or a stop just before (tuned on 59 camera-checked stops, 111 segments).
UNITS_RC = 0.25          # s
BRAKING_MIN = 0.8        # u201, while moving
HOLD_MEMORY = 4.0        # s, from the last braking to the stop
GREEN_MIN = 0.5          # u239, while moving
GREEN_MAX_OTHERS_RATIO = 0.5  # the other units together must stay below this part of u239
MOVING_SPEED = 1.0       # m/s
STOPPED_SPEED = 0.3      # m/s
MODEL_DT = 0.05

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


class TrafficLightClassifier:
  def __init__(self):
    self._units = [FirstOrderFilter(0.0, UNITS_RC, MODEL_DT) for _ in range(4)]
    self._since_braking = float('inf')

  def update(self, units: Sequence[float], v_ego: float) -> TrafficLightState:
    # an empty list means the model does not have the units
    if len(units) != 4:
      return TrafficLightState.NONE

    u69, u248, u201, u239 = (f.update(u) for f, u in zip(self._units, units, strict=True))
    self._since_braking += MODEL_DT
    if v_ego > MOVING_SPEED and u201 > BRAKING_MIN and u201 >= u239:
      self._since_braking = 0.0
      return TrafficLightState.BRAKING
    if v_ego < STOPPED_SPEED and self._since_braking < HOLD_MEMORY:
      # stay red while the car is held
      self._since_braking = 0.0
      return TrafficLightState.HOLD
    if v_ego > MOVING_SPEED and u239 > GREEN_MIN and u69 + u248 + u201 < GREEN_MAX_OTHERS_RATIO * u239:
      return TrafficLightState.GREEN
    return TrafficLightState.NONE


def _with_alpha(color: rl.Color, alpha: float) -> rl.Color:
  return rl.Color(color.r, color.g, color.b, round(color.a * alpha))


class TrafficLight(Widget):
  def __init__(self):
    super().__init__()
    self._alpha_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._classifier = TrafficLightClassifier()
    self._state = TrafficLightState.NONE
    self._shown = TrafficLightState.NONE
    self._should_draw = False

  def set_should_draw(self, should_draw: bool):
    self._should_draw = should_draw

  def _update_state(self):
    super()._update_state()
    # the units come at the model rate, not the UI rate
    if ui_state.sm.updated['modelV2']:
      self._shown = self._classifier.update(ui_state.sm['modelV2'].trafficUnits, ui_state.sm['carState'].vEgo)
    # keep the last color while the icon fades out
    if self._shown != TrafficLightState.NONE:
      self._state = self._shown
    self._alpha_filter.update(float(self._should_draw and self._shown != TrafficLightState.NONE))

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
