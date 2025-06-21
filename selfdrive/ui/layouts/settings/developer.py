from openpilot.system.ui.lib.list_view import ListView, ToggleItem
from openpilot.common.params import Params
from openpilot.selfdrive.ui.widgets.ssh_key import SshKeyItem
from openpilot.system.ui.lib.widget import Widget

# Description constants
DESCRIPTIONS = {
  'enable_adb': (
    "ADB (Android Debug Bridge) allows connecting to your device over USB or over the network. " +
    "See https://docs.comma.ai/how-to/connect-to-comma for more info."
  ),
  'joystick_debug_mode': "Preview the driver facing camera to ensure that driver monitoring has good visibility. (vehicle must be off)",
  'ssh_key': (
    "Warning: This grants SSH access to all public keys in your GitHub settings. Never enter a GitHub username " +
    "other than your own. A comma employee will NEVER ask you to add their GitHub username."
  ),
}


class DeveloperLayout(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()
    items = [
      ToggleItem(
        "Enable ADB",
        DESCRIPTIONS["enable_adb"],
        initial_state=self._params.get_bool("AdbEnabled"),
        callback=self._on_enable_adb,
      ),
      SshKeyItem("SSH Key", description=DESCRIPTIONS["ssh_key"]),
      ToggleItem(
        "Joystick Debug Mode",
        DESCRIPTIONS["joystick_debug_mode"],
        initial_state=self._params.get_bool("JoystickDebugMode"),
        callback=self._on_joystick_debug_mode,
      ),
      ToggleItem(
        "Longitudinal Maneuver Mode",
        "",
        initial_state=self._params.get_bool("LongitudinalManeuverMode"),
        callback=self._on_long_maneuver_mode,
      ),
      ToggleItem(
        "openpilot Longitudinal Control (Alpha)",
        "",
        initial_state=self._params.get_bool("AlphaLongitudinalEnabled"),
        callback=self._on_alpha_long_enabled,
      ),
    ]

    self._list_widget = ListView(items)

  def _render(self, rect):
    self._list_widget.render(rect)

  def _on_enable_adb(self, state): pass
  def _on_joystick_debug_mode(self, state): pass
  def _on_long_maneuver_mode(self, state): pass
  def _on_alpha_long_enabled(self, state): pass
