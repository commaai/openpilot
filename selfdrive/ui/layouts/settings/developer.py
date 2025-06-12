from cereal import car
import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.widgets.ssh_key import SshKeyItem
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.list_view import ListView, ToggleItem
from openpilot.system.ui.lib.widget import Widget, DialogResult
from openpilot.system.ui.widgets.confirm_dialog import confirm_dialog


# Description constants
DESCRIPTIONS = {
  'AdbEnabled': (
    "ADB (Android Debug Bridge) allows connecting to your device over USB or over the network. " +
    "See https://docs.comma.ai/how-to/connect-to-comma for more info."
  ),
  'JoystickDebugMode': "Preview the driver facing camera to ensure that driver monitoring has good visibility. (vehicle must be off)",
  'ssh_key': (
    "Warning: This grants SSH access to all public keys in your GitHub settings. Never enter a GitHub username " +
    "other than your own. A comma employee will NEVER ask you to add their GitHub username."
  ),
  'AlphaLongitudinalEnabled': (
    "WARNING: openpilot longitudinal control is in alpha for this car and will disable Automatic Emergency Braking (AEB).\n" +
    "On this car, openpilot defaults to the car's built-in ACC instead of openpilot's longitudinal control. \n" +
    "Enable this to switch to openpilot longitudinal control. Enabling Experimental mode is recommended when enabling openpilot longitudinal control alpha."
  ),
}


class DeveloperLayout(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()
    self.is_release = self._params.get_bool("IsReleaseBranch")

    self.joystick_toggle = self._make_toggle("Joystick Debug Mode", "JoystickDebugMode",
                                             visible=not self.is_release, enabled=ui_state.is_offroad)
    self.maneuver_toggle = self._make_toggle("Longitudinal Maneuver Mode", "LongitudinalManeuverMode",
                                             visible=not self.is_release, enabled=ui_state.is_offroad)
    self.alpha_long_toggle = self._make_toggle("openpilot Longitudinal Control (Alpha)", "AlphaLongitudinalEnabled",
                                               visible=not self.is_release)

    self._list_view = ListView([
      self._make_toggle("Enable ADB", "AdbEnabled", visible=not self.is_release, enabled=ui_state.is_offroad),
      self._make_toggle("Enable SSH", "SshEnabled"),
      SshKeyItem("SSH Key", description=DESCRIPTIONS["ssh_key"]),
      self.joystick_toggle,
      self.maneuver_toggle,
      self.alpha_long_toggle,
    ])

    self._update_state()

  def _render(self, rect):
    self._list_view.render(rect)

  def _make_toggle(self, title: str, param_name: str, visible=True, enabled=True) -> ToggleItem:
    return ToggleItem(
      title, DESCRIPTIONS.get(param_name, ""),
      initial_state=self._params.get_bool(param_name),
      callback=lambda item: self._on_toggle_change(item, param_name),
      visible = visible,
      enabled = enabled
    )

  def _update_state(self):
    """Update item visibility and state based on car parameters."""
    car_data = self._params.get("CarParamsPersistent")
    if not car_data:
      self.maneuver_toggle.set_enabled(False)
      self.alpha_long_toggle.set_visible(False)
      return

    cp = messaging.log_from_bytes(car_data, car.CarParams)
    alpha_long_available = cp.alphaLongitudinalAvailable

     #Clean up alpha param if not available
    if not alpha_long_available or self.is_release:
      self._params.remove("AlphaLongitudinalEnabled")

    # Update visibility and enabled state
    self.alpha_long_toggle.set_visible(alpha_long_available)
    self.maneuver_toggle.set_enabled(ui_state.is_offroad() and self._is_alpha_long_enabled(cp))

  def _on_toggle_change(self, item: ToggleItem, param_name: str):
     # Special case: alpha long toggle needs confirmation
    if param_name == "AlphaLongitudinalEnabled" and item.get_state():
      gui_app.set_modal_overlay(
        lambda: confirm_dialog(DESCRIPTIONS["AlphaLongitudinalEnabled"], "Enable"),
        callback=self._on_alpha_long_toggle_confirmed,
      )
      return

    # Handle mutually exclusive debug modes
    if param_name == "JoystickDebugMode":
      self._params.put_bool("LongitudinalManeuverMode", False)
      self.maneuver_toggle.set_state(False)
    elif param_name == "LongitudinalManeuverMode":
      self._params.put_bool("JoystickDebugMode", False)
      self.joystick_toggle.set_state(False)

    # Save parameter value
    self._params.put_bool(param_name, item.get_state())

  def _is_alpha_long_enabled(self, cp: car.CarParams):
    if cp.alphaLongitudinalAvailable:
      return self._params.get_bool("AlphaLongitudinalEnabled")
    return cp.openpilotLongitudinalControl

  def _on_alpha_long_toggle_confirmed(self, result: DialogResult):
    enabled = (result == DialogResult.CONFIRM)
    self._params.put_bool("AlphaLongitudinalEnabled", enabled)
    self.alpha_long_toggle.set_state(enabled)
