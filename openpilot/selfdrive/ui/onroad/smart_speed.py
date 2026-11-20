import math
import raylib_nb as rl
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget


class SmartSpeed(Widget):
  def __init__(self):
    super().__init__()
    self._speed_filter = FirstOrderFilter(0.0, 5.0, 1 / gui_app.target_fps, initialized=False)
    self._started_frame = -1
    self._speed = 0
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_medium = gui_app.font(FontWeight.MEDIUM)

  def _update_state(self):
    sm = ui_state.sm
    if self._started_frame != ui_state.started_frame:
      self._speed_filter.initialized = False
      self._started_frame = ui_state.started_frame

    velocity = sm['modelV2'].velocity.x
    if not ui_state.started or sm.recv_frame['modelV2'] < ui_state.started_frame or not sm.valid['modelV2'] or not sm.alive['modelV2'] or not velocity:
      self._speed_filter.initialized = False
      return

    speed = self._speed_filter.update(max(0.0, velocity[-1]))  # terminal prediction at 10 seconds
    conversion = CV.MS_TO_KPH if ui_state.is_metric else CV.MS_TO_MPH
    self._speed = math.ceil(speed * conversion / 5) * 5

  def _render(self, rect: rl.Rectangle):
    if not self._speed_filter.initialized:
      return

    scale = rect.width / 84
    rl.draw_rectangle_rounded(rect, 0.15, 10, rl.WHITE)
    border = rl.Rectangle(rect.x + 3 * scale, rect.y + 3 * scale, rect.width - 6 * scale, rect.height - 6 * scale)
    rl.draw_rectangle_rounded_lines_ex(border, 0.12, 10, 2 * scale, rl.BLACK)

    unit = tr("km/h") if ui_state.is_metric else tr("mph")
    for text, font, size, y in (("smart-speed", self._font_medium, 11, 9),
                               (str(self._speed), self._font_bold, 36, 25),
                               (unit, self._font_medium, 12, 75)):
      width = measure_text_cached(font, text, size * scale).x
      rl.draw_text_ex(font, text, rl.Vector2(rect.x + (rect.width - width) / 2, rect.y + y * scale), size * scale, 0, rl.BLACK)
