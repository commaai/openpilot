import raylib_nb as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import gui_label


class ExperimentalModeButton(Widget):
  HEIGHT = 125
  ICON_SIZE = 80
  PADDING = 25

  def __init__(self):
    super().__init__()
    self._chill_icon = gui_app.texture("icons/couch.png", self.ICON_SIZE, self.ICON_SIZE)
    self._experimental_icon = gui_app.texture("icons/experimental_grey.png", self.ICON_SIZE, self.ICON_SIZE)

  def _render(self, rect: rl.Rectangle):
    alpha = 204 if self.is_pressed else 255
    if ui_state.experimental_mode:
      text, icon = tr("EXPERIMENTAL MODE ON"), self._experimental_icon
      start, end = rl.Color(255, 155, 63, alpha), rl.Color(219, 56, 34, alpha)
    else:
      text, icon = tr("CHILL MODE ON"), self._chill_icon
      start, end = rl.Color(20, 255, 171, alpha), rl.Color(35, 149, 255, alpha)

    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    rl.draw_rectangle_gradient_h(int(rect.x), int(rect.y), int(rect.width), int(rect.height), start, end)
    rl.draw_rectangle_rounded_lines_ex(rect, 0.19, 10, 5, rl.BLACK)
    rl.end_scissor_mode()

    label_width = rect.width - self.ICON_SIZE - 2 * self.PADDING
    line_x = rect.x + label_width
    rl.draw_line_ex(rl.Vector2(line_x, rect.y), rl.Vector2(line_x, rect.y + rect.height), 3, rl.Color(0, 0, 0, 77))
    gui_label(rl.Rectangle(rect.x + self.PADDING, rect.y, label_width - self.PADDING, rect.height), text, 45, rl.BLACK)
    rl.draw_texture_ex(icon, rl.Vector2(line_x + self.PADDING, rect.y + (rect.height - icon.height) / 2), 0.0, 1.0, rl.WHITE)
