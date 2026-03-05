from collections.abc import Callable

from openpilot.common.params import Params
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.mici.widgets.button import BigParamControl, BigButton, BigCircleParamControl, BigCircleToggle
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationDialogV2
from openpilot.system.ui.widgets import NavWidget
from openpilot.selfdrive.ui.ui_state import ui_state


class AsiusLayoutMici(NavWidget):
  def __init__(self, back_callback: Callable):
    super().__init__()
    self.set_back_callback(back_callback)
    self._params = Params()

    # Single BLE action button — shows pairing or remove depending on token state
    self._ble_button = BigButton("ble pairing", "start pairing")
    self._ble_button.set_click_callback(self._start_ble_pairing)

    lane_turn_toggle = BigParamControl("lane turn desire", "LaneTurnDesire")
    coop_steering_toggle = BigParamControl("cooperative steering", "TeslaCoopSteering")

    # Small circle toggles (like SSH/ADB in developer menu)
    self._ble_toggle = BigCircleParamControl(
      "asius/ble_short.png", "EnableBLE", toggle_callback=self._handle_ble_toggle, icon_size=(82, 82), icon_offset=(0, 12)
    )
    self._webrtc_toggle = BigCircleParamControl("asius/webrtc_short.png", "EnableWebRTC", icon_size=(82, 82), icon_offset=(0, 12))
    self._asius_api_toggle = BigCircleToggle("asius/asius_short.png", toggle_callback=self._handle_asius_api_toggle, icon_size=(82, 82), icon_offset=(0, 12))
    self._asius_api_toggle.set_checked(bool(self._params.get("AsiusAPIHost")))

    self._scroller = Scroller(
      [
        self._ble_toggle,
        self._ble_button,
        self._webrtc_toggle,
        self._asius_api_toggle,
        lane_turn_toggle,
        coop_steering_toggle,
      ],
      snap_items=False,
    )

    self._refresh_toggles = (
      ("EnableBLE", self._ble_toggle),
      ("EnableWebRTC", self._webrtc_toggle),
      ("LaneTurnDesire", lane_turn_toggle),
      ("TeslaCoopSteering", coop_steering_toggle),
      ("AsiusAPIHost", self._asius_api_toggle),
    )

  def show_event(self):
    super().show_event()
    self._scroller.show_event()
    self._update_toggles()

  def _update_toggles(self):
    ui_state.update_params()
    for key, item in self._refresh_toggles:
      state = bool(ui_state.params.get(key)) if key == "AsiusAPIHost" else ui_state.params.get_bool(key)
      item.set_checked(state)

    ble_enabled = ui_state.params.get_bool("EnableBLE")

    from openpilot.system.athena.ble import get_ble_token

    has_token = get_ble_token() is not None
    pairing_code = ui_state.params.get("BlePairingCode")

    if has_token:
      self._ble_button.set_text("paired ble device")
      self._ble_button.set_value("remove")
      self._ble_button.set_click_callback(self._remove_ble_device)
    elif pairing_code:
      self._ble_button.set_text("ble pairing code")
      self._ble_button.set_value(pairing_code)
      self._ble_button.set_click_callback(self._stop_ble_pairing)
    else:
      self._ble_button.set_text("ble pairing")
      self._ble_button.set_value("start pairing")
      self._ble_button.set_click_callback(self._start_ble_pairing)

    self._ble_button.set_enabled(ble_enabled)
    self._ble_button.set_visible(ble_enabled)

  def _render(self, rect):
    self._update_toggles()
    self._scroller.render(rect)

  def _start_ble_pairing(self):
    from openpilot.system.athena.ble import start_pairing

    start_pairing()
    self._update_toggles()

  def _stop_ble_pairing(self):
    from openpilot.system.athena.ble import stop_pairing

    stop_pairing()
    self._update_toggles()

  def _remove_ble_device(self):
    from openpilot.system.athena.ble import clear_ble_token

    clear_ble_token()
    self._update_toggles()

  def _handle_ble_toggle(self, state: bool):
    if not state:
      from openpilot.system.athena.ble import stop_pairing

      stop_pairing()
    # param is written by BigCircleParamControl after this callback,
    # so update the BLE button directly using the new state
    self._ble_button.set_enabled(state)
    self._ble_button.set_visible(state)

  def _handle_asius_api_toggle(self, state: bool):
    title = "switch to asius api" if state else "switch to comma api"

    def on_confirm():
      self._params.put("AsiusAPIHost", "api.asius.ai" if state else "")
      self._params.remove("DongleId")
      self._params.put_bool_nonblocking("DoReboot", True)

    dlg = BigConfirmationDialogV2(title, "icons_mici/settings/device/reboot.png", red=True, confirm_callback=on_confirm)
    gui_app.set_modal_overlay(dlg, callback=lambda _: self._asius_api_toggle.set_checked(bool(self._params.get("AsiusAPIHost"))))
