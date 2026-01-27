from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.list_view import toggle_item
from openpilot.system.ui.widgets.scroller_tici import Scroller
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.widgets import DialogResult

DESCRIPTIONS = {
  "EnableAsiusAPI": tr_noop("Use Asius API for Connect features. Disabling will switch to comma API. Requires reboot to re-register device."),
  "EnableWebRTC": tr_noop("Allow remote live streaming via Connect."),
  "EnableRemoteParams": tr_noop("Allow remote parameter editing via Connect."),
  "EnableBLE": tr_noop("Enable Bluetooth Low Energy server for local device control without network."),
  "TeslaCoopSteering": tr_noop(
    "Allows the driver to provide limited steering input while openpilot is engaged. "
    "Warning: May experience steering oscillations below 30 mph during turns."
  ),
}


class AsiusLayout(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()

    self._toggle_defs = {
      "EnableAsiusAPI": (
        lambda: tr("Asius API"),
        DESCRIPTIONS["EnableAsiusAPI"],
        "asius.png",
      ),
      "EnableWebRTC": (
        lambda: tr("Remote Live Streaming"),
        DESCRIPTIONS["EnableWebRTC"],
        "network.png",
      ),
      "EnableRemoteParams": (
        lambda: tr("Remote Parameter Editing"),
        DESCRIPTIONS["EnableRemoteParams"],
        "settings.png",
      ),
      "EnableBLE": (
        lambda: tr("Bluetooth Control"),
        DESCRIPTIONS["EnableBLE"],
        "bluetooth.png",
      ),
      "TeslaCoopSteering": (
        lambda: tr("Tesla Cooperative Steering"),
        DESCRIPTIONS["TeslaCoopSteering"],
        "chffr_wheel.png",
      ),
    }

    self._toggles = {}
    for param, (title, desc, icon) in self._toggle_defs.items():
      toggle = toggle_item(
        title,
        desc,
        self._params.get_bool(param),
        callback=lambda state, p=param: self._toggle_callback(state, p),
        icon=icon,
      )
      self._toggles[param] = toggle

    self._scroller = Scroller(list(self._toggles.values()), line_separator=True, spacing=0)

  def show_event(self):
    self._scroller.show_event()
    self._update_toggles()

  def _update_toggles(self):
    for param in self._toggle_defs:
      self._toggles[param].action_item.set_state(self._params.get_bool(param))

    # Only show Tesla-specific toggles for Tesla vehicles
    is_tesla = ui_state.CP is not None and ui_state.CP.brand == "tesla"
    self._toggles["TeslaCoopSteering"].set_visible(is_tesla)

  def _render(self, rect):
    self._scroller.render(rect)

  def _toggle_callback(self, state: bool, param: str):
    if param == "EnableAsiusAPI":
      self._handle_asius_api_toggle(state)
      return

    self._params.put_bool(param, state)

  def _handle_asius_api_toggle(self, state: bool):
    msg = tr("Switch to Asius API? This will clear your device registration and reboot.") if state \
      else tr("Switch to comma API? This will clear your device registration and reboot.")

    def confirm_callback(result: int):
      if result == DialogResult.CONFIRM:
        self._params.put_bool("EnableAsiusAPI", state)
        self._params.remove("DongleId")
        self._params.put_bool_nonblocking("DoReboot", True)
      else:
        self._toggles["EnableAsiusAPI"].action_item.set_state(not state)

    dlg = ConfirmDialog(msg, tr("Switch and Reboot"))
    gui_app.set_modal_overlay(dlg, callback=confirm_callback)
