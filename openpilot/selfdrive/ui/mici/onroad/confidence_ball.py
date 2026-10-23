import math
import pyray as rl
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.lib.application import gui_app
from openpilot.common.filter_simple import FirstOrderFilter


DOT_RADIUS = 16
DOT_MARGIN = 10


def draw_circle_gradient(center_x: float, center_y: float, radius: int,
                         top: rl.Color, bottom: rl.Color, segments: int = 36) -> None:
  # vertical gradient, the gpu interpolates the vertex colors
  def vertex(angle: float):
    y = center_y + radius * math.sin(angle)
    color = rl.color_lerp(top, bottom, (y - center_y + radius) / (2 * radius))
    rl.rl_color4ub(color.r, color.g, color.b, color.a)
    rl.rl_vertex2f(center_x + radius * math.cos(angle), y)

  center_color = rl.color_lerp(top, bottom, 0.5)
  rl.rl_begin(rl.RL_TRIANGLES)
  for i in range(segments):
    rl.rl_color4ub(center_color.r, center_color.g, center_color.b, center_color.a)
    rl.rl_vertex2f(center_x, center_y)
    vertex(2 * math.pi * (i + 1) / segments)
    vertex(2 * math.pi * i / segments)
  rl.rl_end()


class ConfidenceBall(Widget):
  def __init__(self, demo: bool = False):
    super().__init__()
    self._demo = demo
    self._confidence_filter = FirstOrderFilter(-0.5, 0.5, 1 / gui_app.target_fps)

  def update_filter(self, value: float):
    self._confidence_filter.update(value)

  def _update_state(self):
    if self._demo:
      return

    # animate status dot in from bottom, rest there while disengaged
    if ui_state.status == UIStatus.DISENGAGED:
      self._confidence_filter.update(0.0)
    else:
      self._confidence_filter.update((1 - max(ui_state.sm['modelV2'].meta.disengagePredictions.brakeDisengageProbs or [1])) *
                                                        (1 - max(ui_state.sm['modelV2'].meta.disengagePredictions.steerOverrideProbs or [1])))

  def _render(self, _):
    dot_inset = DOT_MARGIN + DOT_RADIUS
    dot_x = self._rect.x + self._rect.width - dot_inset
    dot_y = self._rect.y + dot_inset + (1 - self._confidence_filter.x) * (self._rect.height - 2 * dot_inset)

    # confidence zones
    if ui_state.status == UIStatus.ENGAGED or self._demo:
      if self._confidence_filter.x > 0.5:
        top_dot_color = rl.Color(0, 255, 204, 255)
        bottom_dot_color = rl.Color(0, 255, 38, 255)
      elif self._confidence_filter.x > 0.2:
        top_dot_color = rl.Color(255, 200, 0, 255)
        bottom_dot_color = rl.Color(255, 115, 0, 255)
      else:
        top_dot_color = rl.Color(255, 0, 21, 255)
        bottom_dot_color = rl.Color(255, 0, 89, 255)

    elif ui_state.status == UIStatus.OVERRIDE:
      top_dot_color = rl.Color(255, 255, 255, 255)
      bottom_dot_color = rl.Color(82, 82, 82, 255)

    else:
      top_dot_color = rl.Color(102, 102, 102, 255)
      bottom_dot_color = rl.Color(26, 26, 26, 255)

    # slight drop shadow
    rl.draw_circle_gradient(rl.Vector2(dot_x, dot_y + 2), DOT_RADIUS + 4, rl.Color(0, 0, 0, 90), rl.BLANK)
    draw_circle_gradient(dot_x, dot_y, DOT_RADIUS, top_dot_color, bottom_dot_color)
