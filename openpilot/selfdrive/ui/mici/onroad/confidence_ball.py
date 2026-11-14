import numpy as np
import raylib_nb as rl
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.selfdrive.ui.mici.onroad.status import status_fade
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.polygon import Gradient, draw_polygon
from openpilot.common.filter_simple import FirstOrderFilter


CIRCLE_PTS = np.stack([np.cos(t := np.linspace(0, 2 * np.pi, 64, endpoint=False)), np.sin(t)], axis=1)
# drop shadow from the design: offset (0, 2), blur 8, black at 50%. Stacked discs at gaussian quantiles approximate the blur
SHADOW_OFFSET, SHADOW_SIGMA, SHADOW_ALPHA = 2, 4, 0.5
SHADOW_QUANTILES = (-1.53, -0.89, -0.49, -0.16, 0.16, 0.49, 0.89, 1.53)
SHADOW_LAYER_COLOR = rl.Color(0, 0, 0, round(255 * (1 - (1 - SHADOW_ALPHA) ** (1 / len(SHADOW_QUANTILES)))))


def draw_circle_gradient(center_x: float, center_y: float, radius: float,
                         top: rl.Color, bottom: rl.Color, alpha: float = 1.0) -> None:
  screen = rl.Rectangle(0, 0, gui_app.width, gui_app.height)
  shadow = rl.Color(0, 0, 0, round(SHADOW_LAYER_COLOR.a * alpha))
  for z in SHADOW_QUANTILES:
    draw_polygon(screen, CIRCLE_PTS * (radius + z * SHADOW_SIGMA) + (center_x, center_y + SHADOW_OFFSET), color=shadow)
  top, bottom = rl.Color(top.r, top.g, top.b, round(top.a * alpha)), rl.Color(bottom.r, bottom.g, bottom.b, round(bottom.a * alpha))
  # gradient y is measured from the bottom of the screen
  gradient = Gradient(start=(0, 1 - (center_y - radius) / gui_app.height), end=(0, 1 - (center_y + radius) / gui_app.height),
                      colors=[bottom, top], stops=[0.0, 1.0])
  draw_polygon(screen, CIRCLE_PTS * radius + (center_x, center_y), gradient=gradient)


class ConfidenceBall(Widget):
  def __init__(self):
    super().__init__()
    self._confidence_filter = FirstOrderFilter(0.0, 0.5, 1 / gui_app.target_fps)
    self._alpha_filter = FirstOrderFilter(1.0, 0.05, 1 / gui_app.target_fps)
    self._hidden = False

  def set_hidden(self, hidden: bool):
    self._hidden = hidden

  def _update_state(self):
    # rests at the bottom while disengaged
    if ui_state.status == UIStatus.DISENGAGED:
      self._confidence_filter.update(0.0)
    else:
      self._confidence_filter.update((1 - max(ui_state.sm['modelV2'].meta.disengagePredictions.brakeDisengageProbs or [1])) *
                                                        (1 - max(ui_state.sm['modelV2'].meta.disengagePredictions.steerOverrideProbs or [1])))

  def _render(self, _):
    alpha = self._alpha_filter.update(0.0 if self._hidden else 1.0)
    if alpha < 1e-2:
      return

    # drawn over the right edge of the camera view
    status_dot_radius = 16
    margin = 10
    dot_x = self._rect.x + self._rect.width - margin - status_dot_radius
    dot_height = (1 - self._confidence_filter.x) * (self._rect.height - 2 * (status_dot_radius + margin)) + status_dot_radius + margin
    dot_height = self._rect.y + dot_height

    # confidence zones while engaged, top and bottom gradient colors
    if self._confidence_filter.x > 0.5:
      engaged = (rl.Color(0, 255, 187, 255), rl.Color(0, 255, 64, 255))
    elif self._confidence_filter.x > 0.2:
      engaged = (rl.Color(255, 200, 0, 255), rl.Color(255, 115, 0, 255))
    else:
      engaged = (rl.Color(255, 0, 21, 255), rl.Color(255, 0, 89, 255))
    override = (rl.Color(255, 255, 255, 255), rl.Color(82, 82, 82, 255))
    disengaged = (rl.Color(102, 102, 102, 255), rl.Color(26, 26, 26, 255))
    top_dot_color, bottom_dot_color = (status_fade.color({UIStatus.ENGAGED: e, UIStatus.OVERRIDE: o, UIStatus.DISENGAGED: d})
                                       for e, o, d in zip(engaged, override, disengaged, strict=True))

    draw_circle_gradient(dot_x, dot_height, status_dot_radius,
                         top_dot_color, bottom_dot_color, alpha)
