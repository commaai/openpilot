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
  "EnableBLE": tr_noop("Make device discoverable via Bluetooth for local control without network."),
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
      "EnableBLE": (
        lambda: tr("Bluetooth"),
        DESCRIPTIONS["EnableBLE"],
        "../asius/bluetooth.png",
      ),
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

      if param == "EnableBLE":
        self._ble_button = self._build_ble_button()
        self._items.append(self._ble_button)

    self._scroller = Scroller(self._items, line_separator=True, spacing=0)

    # Track BLE state so we only rebuild when something changes
    self._prev_ble_state = self._get_ble_state()

  def _get_ble_state(self) -> tuple:
    from openpilot.system.athena.ble import get_ble_token

    ble_enabled = self._params.get_bool("EnableBLE")
    has_token = get_ble_token() is not None
    pairing_code = self._params.get("BlePairingCode")
    return (ble_enabled, has_token, pairing_code)

  def show_event(self):
    self._scroller.show_event()
    self._update_toggles()

  def _update_toggles(self):
    for param in self._toggle_defs:
      state = bool(self._params.get(param)) if param == "AsiusAPIHost" else self._params.get_bool(param)
      self._toggles[param].action_item.set_state(state)

  def _render(self, rect):
    # Poll for BLE state changes and rebuild if needed
    ble_state = self._get_ble_state()
    if ble_state != self._prev_ble_state:
      self._prev_ble_state = ble_state
      self._rebuild_scroller()

    self._scroller.render(rect)

  def _build_ble_button(self):
    from openpilot.system.athena.ble import get_ble_token

    has_token = get_ble_token() is not None
    pairing_code = self._params.get("BlePairingCode")

    if has_token:
      btn = button_item(
        lambda: tr("Paired BLE Device"),
        lambda: tr("Remove"),
        description=lambda: tr("Remove the paired device's access"),
        callback=self._remove_ble_device,
        enabled=lambda: self._params.get_bool("EnableBLE"),
      )
    elif pairing_code:
      btn = button_item(
        lambda: tr("BLE Pairing Code"),
        lambda: self._params.get("BlePairingCode") or "",
        description=lambda: tr("Tap to cancel pairing mode"),
        callback=self._stop_ble_pairing,
        enabled=lambda: self._params.get_bool("EnableBLE"),
      )
    else:
      btn = button_item(
        lambda: tr("BLE Pairing"),
        lambda: tr("Start Pairing"),
        description=lambda: tr("Start pairing mode to connect a device"),
        callback=self._start_ble_pairing,
        enabled=lambda: self._params.get_bool("EnableBLE"),
      )

    btn.set_visible(lambda: self._params.get_bool("EnableBLE"))
    return btn

  def _rebuild_scroller(self):
    self._items = []
    for param in self._toggle_defs:
      self._items.append(self._toggles[param])
      if param == "EnableBLE":
        self._ble_button = self._build_ble_button()
        self._items.append(self._ble_button)
    self._scroller = Scroller(self._items, line_separator=True, spacing=0)

  def _toggle_callback(self, state: bool, param: str):
    if param == "AsiusAPIHost":
      self._handle_asius_api_toggle(state)
      return

    self._params.put_bool(param, state)

    if param == "EnableBLE" and not state:
      from openpilot.system.athena.ble import stop_pairing

      stop_pairing()

    if param == "EnableBLE":
      self._rebuild_scroller()

  def _start_ble_pairing(self):
    from openpilot.system.athena.ble import start_pairing

    start_pairing()
    self._rebuild_scroller()

  def _stop_ble_pairing(self):
    from openpilot.system.athena.ble import stop_pairing

    stop_pairing()
    self._rebuild_scroller()

  def _remove_ble_device(self):
    from openpilot.system.athena.ble import clear_ble_token

    clear_ble_token()
    self._rebuild_scroller()

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
