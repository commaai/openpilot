from openpilot.system.ui.widgets.scroller import NavScroller
from openpilot.selfdrive.ui.mici.widgets.button import BigParamControl, BigCircleParamControl
from openpilot.selfdrive.ui.ui_state import ui_state


class AsiusLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()

    lane_turn_toggle = BigParamControl("lane turn desire", "LaneTurnDesire")
    coop_steering_toggle = BigParamControl("cooperative steering", "TeslaCoopSteering")

    self._webrtc_toggle = BigCircleParamControl("asius/webrtc_short.png", "EnableWebRTC", icon_size=(82, 82), icon_offset=(0, 12))

    self._scroller.add_widgets([
      self._webrtc_toggle,
      lane_turn_toggle,
      coop_steering_toggle,
    ])

    self._refresh_toggles = (
      ("EnableWebRTC", self._webrtc_toggle),
      ("LaneTurnDesire", lane_turn_toggle),
      ("TeslaCoopSteering", coop_steering_toggle),
    )

  def show_event(self):
    super().show_event()
    self._update_toggles()

  def _update_toggles(self):
    ui_state.update_params()
    for key, item in self._refresh_toggles:
      item.set_checked(ui_state.params.get_bool(key))

  def _render(self, rect):
    self._update_toggles()
    super()._render(rect)

