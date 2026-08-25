import pyray as rl
import threading
from pathlib import Path
from collections.abc import Callable

from openpilot.common.hardware.usb import is_current_chestnut_firmware
from openpilot.common.file_chunker import get_manifest_path
from openpilot.selfdrive.modeld.helpers import modeld_pkl_path
from openpilot.selfdrive.ui.mici.widgets.button import BigButton
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationDialog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.application import MousePos
from openpilot.system.hardware.chestnut.diagnostics import check_chestnut
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets.scroller import NavScroller


class ChestnutInfoLayout(Widget):
  def __init__(self, summary: Callable[[], tuple[str, str, str, str]]):
    super().__init__()
    self._summary = summary
    self.set_rect(rl.Rectangle(0, 0, 360, 180))
    color = rl.Color(255, 255, 255, int(255 * 0.9 * 0.65))
    max_width = int(self._rect.width - 20)
    self._status = UnifiedLabel("", 48, max_width=max_width, font_weight=FontWeight.DISPLAY, wrap_text=False)
    self._status_value = UnifiedLabel("", 32, max_width=max_width, text_color=color, font_weight=FontWeight.ROMAN, wrap_text=False)
    self._connection = UnifiedLabel("", 48, max_width=max_width, font_weight=FontWeight.DISPLAY, wrap_text=False)
    self._issue_value = UnifiedLabel("", 32, max_width=max_width, text_color=color, font_weight=FontWeight.ROMAN, wrap_text=False)

  def _update_state(self):
    title, status, connection, connection_value = self._summary()
    self._status.set_text(title)
    self._status_value.set_text(status)
    self._connection.set_text(connection)
    self._issue_value.set_text(connection_value)

  def _render(self, _):
    x, y = self._rect.x + 20, self._rect.y
    for label, offset in ((self._status, -10), (self._status_value, 43), (self._connection, 84), (self._issue_value, 136)):
      label.set_position(x, y + offset)
      label.render()


class CheckChestnutButton(BigButton):
  def __init__(self):
    self._check_icon = gui_app.texture("icons_mici/settings/device/cable.png", 64, 64)
    self._success_icon = gui_app.texture("icons_mici/settings/device/up_to_date.png", 64, 64)
    self._error_icon = gui_app.texture("icons_mici/setup/warning.png", 64, 58)
    super().__init__("check connection", "", self._check_icon)
    self._running = False
    self._done = False
    self._result: str | None = None
    self._hide_value_t: float | None = None

  def set_value(self, value: str):
    super().set_value(value)
    self.set_text("" if value else "check connection")

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)
    if ui_state.started or self._running:
      return

    self._running = True
    self._done = False
    self._result = None
    self.set_enabled(False)
    def run():
      try:
        self._result = check_chestnut()
      finally:
        self._done = True

    threading.Thread(target=run, daemon=True).start()

  def _update_state(self):
    super()._update_state()
    if self._running:
      self.set_rotate_icon(True)
      self.set_value("checking...")
      if self._done:
        self._running = False
        self._hide_value_t = rl.get_time()
        self.set_rotate_icon(False)
        self.set_icon(self._success_icon if self._result is None else self._error_icon)
        self.set_value("no errors" if self._result is None else self._result)
    elif self._hide_value_t is not None and rl.get_time() - self._hide_value_t > 3.:
      self._hide_value_t = None
      self.set_icon(self._check_icon)
      self.set_value("")
    self.set_enabled(not ui_state.started and not self._running)


class ChestnutLayout(NavScroller):
  def __init__(self):
    super().__init__()

    self._scroller.add_widgets([
      ChestnutInfoLayout(self._summary),
      CheckChestnutButton(),
      self._compile_button(),
    ])

  def _compile_button(self):
    button = BigButton("compile model", "keep ignition on", gui_app.texture("icons_mici/settings/device/reboot.png", 64, 70))
    button.set_visible(lambda: not ui_state.chestnut_release and not ui_state.usbgpu_compiled)
    button.set_enabled(lambda: not ui_state.started)
    button.set_click_callback(self._force_compile)
    return button

  def _connection(self):
    device = ui_state.chestnut_device
    if device is None:
      return "not enumerated"
    speed = device.speedMbps
    return f"{speed // 1000} Gbps" if speed >= 1000 and speed % 1000 == 0 else f"{speed} Mbps"

  def _status(self):
    if not ui_state.chestnut_release:
      return "ready" if ui_state.usbgpu_compiled else "not ready (uncompiled)"
    device = ui_state.chestnut_device
    if device is None:
      return "not ready (missing)"
    if device.speedMbps < 5000:
      return "not ready (slow USB)"
    if not is_current_chestnut_firmware(device.product):
      return "not ready (firmware)"
    if not ui_state.chestnut_pcie_connected:
      return "not ready (no 12V)"
    if not ui_state.usbgpu_compiled:
      return "not ready (uncompiled)"
    return "ready"

  def _summary(self):
    return "status", self._status(), "USB link", self._connection()

  @staticmethod
  def _force_compile():
    def reboot():
      Path(get_manifest_path(modeld_pkl_path(usbgpu=True))).unlink(missing_ok=True)
      ui_state.usbgpu_compiled = False
      ui_state.params.put_bool("DoReboot", True, block=True)

    icon = gui_app.texture("icons_mici/settings/device/reboot.png", 64, 70)
    gui_app.push_widget(BigConfirmationDialog("slide to reboot and compile", icon, reboot, exit_on_confirm=False))
