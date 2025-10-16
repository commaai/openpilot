import time
import pyray as rl
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets import DialogResult


class ExpButton(Widget):
  def __init__(self, button_size: int, icon_size: int):
    super().__init__()
    self._params = Params()
    self._experimental_mode: bool = False
    self._engageable: bool = False

    # State hold mechanism
    self._hold_duration = 2.0  # seconds
    self._held_mode: bool | None = None
    self._hold_end_time: float | None = None

    self._white_color: rl.Color = rl.Color(255, 255, 255, 255)
    self._black_bg: rl.Color = rl.Color(0, 0, 0, 166)
    self._txt_wheel: rl.Texture = gui_app.texture('icons/chffr_wheel.png', icon_size, icon_size)
    self._txt_exp: rl.Texture = gui_app.texture('icons/experimental.png', icon_size, icon_size)
    self._rect = rl.Rectangle(0, 0, button_size, button_size)

  def set_rect(self, rect: rl.Rectangle) -> None:
    # Use base implementation to ensure width/height are applied and layout is updated
    super().set_rect(rect)

  def _update_state(self) -> None:
    selfdrive_state = ui_state.sm["selfdriveState"]
    self._experimental_mode = selfdrive_state.experimentalMode
    self._engageable = selfdrive_state.engageable or selfdrive_state.enabled

  def _handle_mouse_release(self, _):
    super()._handle_mouse_release(_)
    if self._is_toggle_allowed():
      new_mode = not self._experimental_mode
      self._params.put_bool("ExperimentalMode", new_mode)

      # Hold new state temporarily
      self._held_mode = new_mode
      self._hold_end_time = time.monotonic() + self._hold_duration
    else:
      # Provide feedback if toggle is blocked due to longitudinal requirements
      if self._params.get_bool("ExperimentalModeConfirmed") and not ui_state.has_longitudinal_control:
        if ui_state.CP and ui_state.CP.alphaLongitudinalAvailable:
          content = ("<h1>Experimental Mode</h1><br>"
                     "<p>Enable the openpilot longitudinal control (alpha) toggle to allow Experimental mode.</p>")
        else:
          content = ("<h1>Experimental Mode</h1><br>"
                     "<p>Experimental mode is unavailable on this car because stock ACC is used for longitudinal control.</p>")
        dlg = ConfirmDialog(content, "OK", rich=True)
        gui_app.set_modal_overlay(dlg, callback=lambda result: gui_app.set_modal_overlay(None))

  def _render(self, rect: rl.Rectangle) -> None:
    center_x = int(self._rect.x + self._rect.width // 2)
    center_y = int(self._rect.y + self._rect.height // 2)

    self._white_color.a = 180 if self.is_pressed or not self._engageable else 255

    texture = self._txt_exp if self._held_or_actual_mode() else self._txt_wheel
    rl.draw_circle(center_x, center_y, self._rect.width / 2, self._black_bg)
    rl.draw_texture(texture, center_x - texture.width // 2, center_y - texture.height // 2, self._white_color)

  def _held_or_actual_mode(self):
    now = time.monotonic()
    if self._hold_end_time and now < self._hold_end_time:
      return self._held_mode

    if self._hold_end_time and now >= self._hold_end_time:
      self._hold_end_time = self._held_mode = None

    return self._experimental_mode

  def _is_toggle_allowed(self):
    if not self._params.get_bool("ExperimentalModeConfirmed"):
      return False

    # Mirror settings logic using computed flag from UI state
    return ui_state.has_longitudinal_control
