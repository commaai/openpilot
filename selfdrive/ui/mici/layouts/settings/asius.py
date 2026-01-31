from collections.abc import Callable

from openpilot.common.params import Params
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.mici.widgets.button import BigParamControl, BigToggle, BigButton
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationDialogV2
from openpilot.system.ui.widgets import NavWidget
from openpilot.selfdrive.ui.ui_state import ui_state


class AsiusLayoutMici(NavWidget):
  def __init__(self, back_callback: Callable):
    super().__init__()
    self.set_back_callback(back_callback)
    self._params = Params()

    asius_api_toggle = BigToggle("asius api", toggle_callback=self._handle_asius_api_toggle)
    asius_api_toggle.set_checked(self._params.get_bool("EnableAsiusAPI"))
    self._asius_api_toggle = asius_api_toggle

    webrtc_toggle = BigParamControl("remote live streaming", "EnableWebRTC")
    remote_params_toggle = BigParamControl("remote parameter editing", "EnableRemoteParams")
    ble_toggle = BigParamControl("bluetooth", "EnableBLE", toggle_callback=self._handle_ble_toggle)

    pairing_code = self._params.get("BlePairingCode")
    if pairing_code:
      self._ble_pairing_button = BigButton("ble pairing code")
      self._ble_pairing_button.set_enabled(False)
      self._ble_pairing_button.set_value(pairing_code)
    else:
      self._ble_pairing_button = BigButton("ble pairing", "start pairing")
      self._ble_pairing_button.set_click_callback(self._start_ble_pairing)

    self._scroller = Scroller([
      ble_toggle,
      self._ble_pairing_button,
      webrtc_toggle,
      remote_params_toggle,
      asius_api_toggle,
    ], snap_items=False)

    self._refresh_toggles = (
      ("EnableBLE", ble_toggle),
      ("EnableWebRTC", webrtc_toggle),
      ("EnableRemoteParams", remote_params_toggle),
      ("EnableAsiusAPI", asius_api_toggle),
    )

  def show_event(self):
    super().show_event()
    self._scroller.show_event()
    self._update_toggles()

  def _update_toggles(self):
    ui_state.update_params()
    for key, item in self._refresh_toggles:
      item.set_checked(ui_state.params.get_bool(key))

    # Update BLE pairing button
    ble_enabled = ui_state.params.get_bool("EnableBLE")
    pairing_code = ui_state.params.get("BlePairingCode")

    if pairing_code:
      self._ble_pairing_button.set_text("ble pairing code")
      self._ble_pairing_button.set_value(pairing_code)
      self._ble_pairing_button.set_enabled(ble_enabled)
      self._ble_pairing_button.set_click_callback(self._stop_ble_pairing)
    else:
      self._ble_pairing_button.set_text("ble pairing")
      self._ble_pairing_button.set_value("start pairing")
      self._ble_pairing_button.set_enabled(ble_enabled)
      self._ble_pairing_button.set_click_callback(self._start_ble_pairing)

  def _render(self, rect):
    self._scroller.render(rect)

  def _start_ble_pairing(self):
    from openpilot.system.athena.ble import start_pairing
    start_pairing()
    self._update_toggles()

  def _stop_ble_pairing(self):
    from openpilot.system.athena.ble import stop_pairing
    stop_pairing()
    self._update_toggles()

  def _handle_ble_toggle(self, state: bool):
    if not state:
      from openpilot.system.athena.ble import stop_pairing
      stop_pairing()

  def _handle_asius_api_toggle(self, state: bool):
    title = "switch to asius api" if state else "switch to comma api"

    def on_confirm():
      self._params.put_bool("EnableAsiusAPI", state)
      self._params.remove("DongleId")
      self._params.put_bool_nonblocking("DoReboot", True)

    dlg = BigConfirmationDialogV2(title, "icons_mici/settings/device/reboot.png", red=True, confirm_callback=on_confirm)
    gui_app.set_modal_overlay(dlg, callback=lambda _: self._asius_api_toggle.set_checked(self._params.get_bool("EnableAsiusAPI")))
