from collections.abc import Callable

from openpilot.common.params import Params
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.mici.widgets.button import BigParamControl, BigToggle
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
    ble_toggle = BigParamControl("bluetooth control", "EnableBLE")

    self._scroller = Scroller([
      ble_toggle,
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

  def _render(self, rect):
    self._scroller.render(rect)

  def _handle_asius_api_toggle(self, state: bool):
    title = "switch to asius api" if state else "switch to comma api"

    def on_confirm():
      self._params.put_bool("EnableAsiusAPI", state)
      self._params.remove("DongleId")
      self._params.put_bool_nonblocking("DoReboot", True)

    dlg = BigConfirmationDialogV2(title, "icons_mici/settings/device/reboot.png", red=True, confirm_callback=on_confirm)
    gui_app.set_modal_overlay(dlg, callback=lambda _: self._asius_api_toggle.set_checked(self._params.get_bool("EnableAsiusAPI")))
