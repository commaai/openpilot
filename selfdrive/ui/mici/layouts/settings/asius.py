from openpilot.common.params import Params
from openpilot.system.ui.widgets.scroller import NavScroller
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.mici.widgets.button import BigParamControl, BigCircleParamControl, BigCircleToggle
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationDialogV2
from openpilot.selfdrive.ui.ui_state import ui_state


class AsiusLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()
    self._params = Params()

    lane_turn_toggle = BigParamControl("lane turn desire", "LaneTurnDesire")
    coop_steering_toggle = BigParamControl("cooperative steering", "TeslaCoopSteering")

    self._webrtc_toggle = BigCircleParamControl("asius/webrtc_short.png", "EnableWebRTC", icon_size=(82, 82), icon_offset=(0, 12))
    self._asius_api_toggle = BigCircleToggle("asius/asius_short.png", toggle_callback=self._handle_asius_api_toggle, icon_size=(82, 82), icon_offset=(0, 12))
    self._asius_api_toggle.set_checked(bool(self._params.get("AsiusAPIHost")))

    self._scroller.add_widgets([
      self._webrtc_toggle,
      self._asius_api_toggle,
      lane_turn_toggle,
      coop_steering_toggle,
    ])

    self._refresh_toggles = (
      ("EnableWebRTC", self._webrtc_toggle),
      ("LaneTurnDesire", lane_turn_toggle),
      ("TeslaCoopSteering", coop_steering_toggle),
      ("AsiusAPIHost", self._asius_api_toggle),
    )

  def show_event(self):
    super().show_event()
    self._update_toggles()

  def _update_toggles(self):
    ui_state.update_params()
    for key, item in self._refresh_toggles:
      state = bool(ui_state.params.get(key)) if key == "AsiusAPIHost" else ui_state.params.get_bool(key)
      item.set_checked(state)

  def _render(self, rect):
    self._update_toggles()
    super()._render(rect)

  def _handle_asius_api_toggle(self, state: bool):
    title = "switch to asius api" if state else "switch to comma api"

    def on_confirm():
      self._params.put("AsiusAPIHost", "api.asius.ai" if state else "")
      self._params.remove("DongleId")
      self._params.put_bool_nonblocking("DoReboot", True)

    dlg = BigConfirmationDialogV2(title, "icons_mici/settings/device/reboot.png", red=True, confirm_callback=on_confirm)
    gui_app.set_modal_overlay(dlg, callback=lambda _: self._asius_api_toggle.set_checked(bool(self._params.get("AsiusAPIHost"))))
