import time
import pyray as rl
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget


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
    self._rect.x, self._rect.y = rect.x, rect.y

  def _update_state(self) -> None:
    selfdrive_state = ui_state.sm["selfdriveState"]
    self._experimental_mode = selfdrive_state.experimentalMode
    self._engageable = selfdrive_state.engageable or selfdrive_state.enabled

  def handle_mouse_event(self) -> bool:
    if rl.check_collision_point_rec(rl.get_mouse_position(), self._rect):
      if (rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT) and
          self._is_toggle_allowed()):
        new_mode = not self._experimental_mode
        self._params.put_bool("ExperimentalMode", new_mode)

        # Hold new state temporarily
        self._held_mode = new_mode
        self._hold_end_time = time.monotonic() + self._hold_duration
      return True
    return False

  def _render(self, rect: rl.Rectangle) -> None:
    center_x = int(self._rect.x + self._rect.width // 2)
    center_y = int(self._rect.y + self._rect.height // 2)

    mouse_over = rl.check_collision_point_rec(rl.get_mouse_position(), self._rect)
    mouse_down = rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT) and self.is_pressed
    self._white_color.a = 180 if (mouse_down and mouse_over) or not self._engageable else 255

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

    car_params = ui_state.sm["carParams"]
    if car_params.alphaLongitudinalAvailable:
      return self._params.get_bool("AlphaLongitudinalEnabled")
    else:
      return car_params.openpilotLongitudinalControl
