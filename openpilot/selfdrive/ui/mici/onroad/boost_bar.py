import pyray as rl
from openpilot.selfdrive.controls.lib.drive_helpers import ACCEL_BOOST_MAX
from openpilot.selfdrive.ui.mici.onroad import SIDE_PANEL_WIDTH
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.widgets import Widget


class BoostBar(Widget):
  def _render(self, rect: rl.Rectangle):
    bar_width = 24
    bar_rect = rl.Rectangle(rect.x + rect.width - (SIDE_PANEL_WIDTH + bar_width) / 2,
                            rect.y + 24, bar_width, rect.height - 48)
    rl.draw_rectangle_rounded(bar_rect, 1.0, 10, rl.Color(50, 50, 50, 255))

    boost = ui_state.sm['longitudinalPlan'].accelBoost if ui_state.engaged else 0.0
    if boost > 0.0:
      fill_height = bar_rect.height * boost / ACCEL_BOOST_MAX
      fill_rect = rl.Rectangle(bar_rect.x, bar_rect.y + bar_rect.height - fill_height, bar_width, fill_height)
      rl.draw_rectangle_rounded(fill_rect, 1.0, 10, rl.Color(0, 255, 204, 255))
