import pyray as rl
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app

SIDE_PANEL_WIDTH = 72


def blend_colors(a: rl.Color, b: rl.Color, f: float) -> rl.Color:
  h0, s0, v0 = (hsv0 := rl.color_to_hsv(a)).x, hsv0.y, hsv0.z
  h1, s1, v1 = (hsv1 := rl.color_to_hsv(b)).x, hsv1.y, hsv1.z
  dh = ((h1 - h0 + 180) % 360) - 180  # shortest hue delta
  return rl.color_from_hsv((h0 + f * dh) % 360,
                           s0 + f * (s1 - s0),
                           v0 + f * (v1 - v0))


class IconIntro:
  # after ignition on, the side panel icons rise in one by one from the bottom once the onroad view slid in
  WHEEL, BOTTOM_BAR, MIDDLE_BAR, TOP_BAR, CAR, DM = range(6)
  STEP = 0.08  # seconds between icons
  RISE = 26  # pixels

  def __init__(self):
    self._filters = [FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps) for _ in range(6)]
    self._started_frame = -1
    self._start_time: float | None = None

  def update(self, in_view: bool) -> None:
    if ui_state.started_frame != self._started_frame:
      self._started_frame = ui_state.started_frame
      self._start_time = None
      for f in self._filters:
        f.x = 0.0

    if self._start_time is None and in_view:
      self._start_time = rl.get_time()

    for i, f in enumerate(self._filters):
      f.update(float(self._start_time is not None and rl.get_time() - self._start_time >= i * self.STEP))

  def alpha(self, icon: int) -> float:
    return self._filters[icon].x

  def offset(self, icon: int) -> float:
    return (1 - self._filters[icon].x) * self.RISE
