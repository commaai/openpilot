from collections.abc import Callable

from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.selfdrive.ui.mici.widgets.button import BigParamControl
from openpilot.system.ui.widgets import NavWidget
from openpilot.selfdrive.ui.ui_state import ui_state


class AsiusLayoutMici(NavWidget):
  def __init__(self, back_callback: Callable):
    super().__init__()
    self.set_back_callback(back_callback)

    asius_api_toggle = BigParamControl("asius api", "EnableAsiusAPI")
    webrtc_toggle = BigParamControl("remote live streaming", "EnableWebRTC")
    remote_params_toggle = BigParamControl("remote parameter editing", "EnableRemoteParams")
    ble_toggle = BigParamControl("bluetooth control", "EnableBLE")

    self._scroller = Scroller([
      asius_api_toggle,
      webrtc_toggle,
      remote_params_toggle,
      ble_toggle,
    ], snap_items=False)

    self._refresh_toggles = (
      ("EnableAsiusAPI", asius_api_toggle),
      ("EnableWebRTC", webrtc_toggle),
      ("EnableRemoteParams", remote_params_toggle),
      ("EnableBLE", ble_toggle),
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
