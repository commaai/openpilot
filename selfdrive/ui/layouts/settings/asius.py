from openpilot.common.params import Params
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.list_view import toggle_item, button_item
from openpilot.system.ui.widgets.scroller_tici import Scroller
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.widgets import DialogResult

DESCRIPTIONS = {
  "AsiusAPIHost": tr_noop("API host for Connect features (e.g. 'api.asius.ai'). Leave empty for comma API. Requires reboot to re-register device."),
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
      "AsiusAPIHost": (
        lambda: tr("API Host"),
        DESCRIPTIONS["AsiusAPIHost"],
        "../asius/asius.png",
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
      initial_state = bool(self._params.get(param)) if param == "AsiusAPIHost" else self._params.get_bool(param)
      toggle = toggle_item(
        title,
        desc,
        initial_state,
        callback=lambda state, p=param: self._toggle_callback(state, p),
        icon=icon,
      )
      self._toggles[param] = toggle
      self._items.append(toggle)

    self._scroller = Scroller(self._items, line_separator=True, spacing=0)

  def show_event(self):
    self._scroller.show_event()
    self._update_toggles()

  def _update_toggles(self):
    for param in self._toggle_defs:
      state = bool(self._params.get(param)) if param == "AsiusAPIHost" else self._params.get_bool(param)
      self._toggles[param].action_item.set_state(state)

  def _render(self, rect):
    self._scroller.render(rect)

  def _toggle_callback(self, state: bool, param: str):
    if param == "AsiusAPIHost":
      self._handle_asius_api_toggle(state)
      return

    self._params.put_bool(param, state)

  def _handle_asius_api_toggle(self, state: bool):
    msg = (
      tr("Switch to Asius API? This will clear your device registration and reboot.")
      if state
      else tr("Switch to comma API? This will clear your device registration and reboot.")
    )

    def confirm_callback(result: int):
      if result == DialogResult.CONFIRM:
        self._params.put("AsiusAPIHost", "api.asius.ai" if state else "")
        self._params.remove("DongleId")
        self._params.put_bool_nonblocking("DoReboot", True)
      else:
        self._toggles["AsiusAPIHost"].action_item.set_state(not state)

    dlg = ConfirmDialog(msg, tr("Switch and Reboot"))
    gui_app.set_modal_overlay(dlg, callback=confirm_callback)
