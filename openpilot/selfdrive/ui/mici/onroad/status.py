"""
Engaged and override can flap every frame (gas taps, steer override on driver torque near the threshold),
so anything that looks different between them must blend through status_fade instead of switching on
ui_state.status.
"""
import numpy as np
import raylib_nb as rl
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.lib.application import gui_app

STATUS_FADE_RC = 0.25
STATUSES = tuple(UIStatus)


def lerp_color(a: rl.Color, b: rl.Color, t: float) -> rl.Color:
  return rl.Color(*(round(x + (y - x) * t) for x, y in zip((a.r, a.g, a.b, a.a), (b.r, b.g, b.b, b.a), strict=True)))


class StatusFade:
  """ui_state.status as weights that fade together, stepped once per frame"""
  def __init__(self):
    self._weights: FirstOrderFilter | None = None
    self._frame = -1

  def _update(self) -> np.ndarray:
    target = np.array([ui_state.status == s for s in STATUSES], dtype=float)
    if self._weights is None:
      self._weights = FirstOrderFilter(target, STATUS_FADE_RC, 1 / gui_app.target_fps)
    elif ui_state.sm.frame != self._frame:
      self._weights.update(target)
    self._frame = ui_state.sm.frame
    return self._weights.x

  def weight(self, status: UIStatus) -> float:
    return float(self._update()[STATUSES.index(status)])

  def color(self, colors: dict[UIStatus, rl.Color]) -> rl.Color:
    rgba = sum(w * np.array([c.r, c.g, c.b, c.a], dtype=float) for w, c in zip(self._update(), (colors[s] for s in STATUSES), strict=True))
    return rl.Color(*np.round(rgba).astype(int).tolist())


status_fade = StatusFade()
