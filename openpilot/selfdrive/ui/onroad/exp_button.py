import raylib_nb as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget


class ExpButton(Widget):
  def __init__(self):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, 192, 192))
    self._wheel = gui_app.texture("icons/chffr_wheel.png", 144, 144)
    self._experimental = gui_app.texture("icons/experimental.png", 144, 144)
    self.set_click_callback(ui_state.toggle_experimental_mode)

  def _render(self, rect: rl.Rectangle):
    state = ui_state.sm["selfdriveState"]
    icon = self._experimental if state.experimentalMode else self._wheel
    tint = rl.Color(255, 255, 255, 180) if self.is_pressed or not (state.engageable or state.enabled) else rl.WHITE
    x, y = rect.x + rect.width / 2, rect.y + rect.height / 2
    rl.draw_circle(int(x), int(y), rect.width / 2, rl.Color(0, 0, 0, 166))
    rl.draw_texture_ex(icon, rl.Vector2(x - icon.width / 2, y - icon.height / 2), 0.0, 1.0, tint)
