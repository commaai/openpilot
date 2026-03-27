from openpilot.common.params import Params
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.list_view import toggle_item
from openpilot.system.ui.widgets.scroller_tici import Scroller
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr, tr_noop

ASIUS_API_HOST = "https://api.asius.ai"
ASIUS_ATHENA_HOST = "wss://api.asius.ai"

DESCRIPTIONS = {
  "EnableWebRTC": tr_noop("Allow remote live streaming via Connect."),
  "LaneTurnDesire": tr_noop("When blinker is on below 20 mph, steer in blinker direction. Useful at intersections and red lights."),
  "TeslaCoopSteering": tr_noop(
    "Allows the driver to provide limited steering input while openpilot is engaged. Blends driver torque into openpilot's steering angle."
  ),
}


class AsiusLayout(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()

    self._toggle_defs = {
      "EnableWebRTC": (
        lambda: tr("Remote Live Streaming"),
        DESCRIPTIONS["EnableWebRTC"],
        "network.png",
      ),
      "LaneTurnDesire": (
        lambda: tr("Lane Turn Desire"),
        DESCRIPTIONS["LaneTurnDesire"],
        "chffr_wheel.png",
      ),
      "TeslaCoopSteering": (
        lambda: tr("Tesla Cooperative Steering"),
        DESCRIPTIONS["TeslaCoopSteering"],
        "chffr_wheel.png",
      ),
    }

    self._toggles = {}
    self._items = []

    for param, (title, desc, icon) in self._toggle_defs.items():
      initial_state = self._params.get_bool(param)
      toggle = toggle_item(
        title,
        desc,
        initial_state,
        callback=lambda state, p=param: self._params.put_bool(p, state),
        icon=icon,
      )
      self._toggles[param] = toggle
      self._items.append(toggle)

    is_asius = self._params.get("APIHost") == ASIUS_API_HOST
    self._api_toggle = toggle_item(
      lambda: tr("Asius API"),
      tr_noop("Use Asius API instead of comma. Requires reboot."),
      is_asius,
      callback=self._handle_api_toggle,
      icon="../asius/asius.png",
    )
    self._items.append(self._api_toggle)

    self._scroller = Scroller(self._items, line_separator=True, spacing=0)

  def show_event(self):
    self._scroller.show_event()
    self._update_toggles()

  def _update_toggles(self):
    for param in self._toggle_defs:
      state = self._params.get_bool(param)
      self._toggles[param].action_item.set_state(state)
    self._api_toggle.action_item.set_state(self._params.get("APIHost") == ASIUS_API_HOST)

  def _render(self, rect):
    self._scroller.render(rect)

  def _handle_api_toggle(self, state: bool):
    msg = tr("Switch to Asius API and reboot?") if state else tr("Switch to comma API and reboot?")

    def confirm_callback(result: int):
      if result == DialogResult.CONFIRM:
        if state:
          self._params.put("APIHost", ASIUS_API_HOST)
          self._params.put("AthenaHost", ASIUS_ATHENA_HOST)
        else:
          self._params.remove("APIHost")
          self._params.remove("AthenaHost")
        self._params.remove("DongleId")
        self._params.put_bool_nonblocking("DoReboot", True)
      else:
        self._api_toggle.action_item.set_state(not state)

    gui_app.set_modal_overlay(ConfirmDialog(msg, tr("Switch and Reboot")), callback=confirm_callback)
