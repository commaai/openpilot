import math
from collections import deque
import pyray as rl
from openpilot.selfdrive.ui.mici.onroad import SIDE_PANEL_WIDTH
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.lib.application import gui_app
from openpilot.common.filter_simple import FirstOrderFilter

TRAIL_LENGTH = 8
TRAIL_INTERVAL = 0.04  # seconds between trail snapshots


def draw_circle_gradient(center_x: float, center_y: float, radius: int,
                         top: rl.Color, bottom: rl.Color) -> None:
  # Draw a square with the gradient
  rl.draw_rectangle_gradient_v(int(center_x - radius), int(center_y - radius),
                               radius * 2, radius * 2,
                               top, bottom)

  # Paint over square with a ring
  outer_radius = math.ceil(radius * math.sqrt(2)) + 1
  rl.draw_ring(rl.Vector2(int(center_x), int(center_y)), radius, outer_radius,
               0.0, 360.0,
               20, rl.BLACK)


class ConfidenceBall(Widget):
  def __init__(self, demo: bool = False):
    super().__init__()
    self._demo = demo
    self._confidence_filter = FirstOrderFilter(-0.5, 0.5, 1 / gui_app.target_fps)

    # Trail state: deque of (dot_x, dot_y, top_color, bottom_color, time)
    self._trail: deque[tuple[float, float, rl.Color, rl.Color, float]] = deque(maxlen=TRAIL_LENGTH)
    self._last_trail_time = 0.0

  def update_filter(self, value: float):
    self._confidence_filter.update(value)

  def _update_state(self):
    if self._demo:
      return

    # animate status dot in from bottom
    if ui_state.status == UIStatus.DISENGAGED:
      self._confidence_filter.update(-0.5)
    else:
      self._confidence_filter.update((1 - max(ui_state.sm['modelV2'].meta.disengagePredictions.brakeDisengageProbs or [1])) *
                                                        (1 - max(ui_state.sm['modelV2'].meta.disengagePredictions.steerOverrideProbs or [1])))

  def _render(self, _):
    content_rect = rl.Rectangle(
      self.rect.x + self.rect.width - SIDE_PANEL_WIDTH,
      self.rect.y,
      SIDE_PANEL_WIDTH,
      self.rect.height,
    )

    base_radius = 24
    pulse = 1.0 + 0.06 * math.sin(rl.get_time() * 4.0)
    status_dot_radius = int(base_radius * pulse)
    dot_height = (1 - self._confidence_filter.x) * (content_rect.height - 2 * status_dot_radius) + status_dot_radius
    dot_height = self._rect.y + dot_height
    dot_x = content_rect.x + content_rect.width - status_dot_radius

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
      top_dot_color = rl.Color(50, 50, 50, 255)
      bottom_dot_color = rl.Color(13, 13, 13, 255)

    # Record trail snapshot
    now = rl.get_time()
    if now - self._last_trail_time >= TRAIL_INTERVAL:
      self._trail.append((dot_x, dot_height, top_dot_color, bottom_dot_color, now))
      self._last_trail_time = now

    # Draw trail as simple faded circles (no gradient to avoid black ring artifact)
    for idx, (tx, ty, tt, tb, t) in enumerate(self._trail):
      age_frac = (idx + 1) / (len(self._trail) + 1)
      trail_alpha = age_frac * 0.35
      trail_radius = status_dot_radius * (0.3 + 0.5 * age_frac)
      # Blend top and bottom colors for a single trail color
      blend_r = (tt.r + tb.r) // 2
      blend_g = (tt.g + tb.g) // 2
      blend_b = (tt.b + tb.b) // 2
      rl.draw_circle(int(tx), int(ty), trail_radius, rl.Color(blend_r, blend_g, blend_b, int(255 * trail_alpha)))

    # Draw main ball
    draw_circle_gradient(dot_x, dot_height, status_dot_radius,
                         top_dot_color, bottom_dot_color)
