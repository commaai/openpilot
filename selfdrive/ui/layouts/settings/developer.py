from openpilot.system.ui.lib.list_view import toggle_item
from openpilot.system.ui.lib.scroller import Scroller
from openpilot.system.ui.lib.widget import Widget
from openpilot.common.params import Params
from openpilot.selfdrive.ui.widgets.ssh_key import ssh_key_item

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
      toggle_item(
        "Enable ADB",
        description=DESCRIPTIONS["enable_adb"],
        initial_state=self._params.get_bool("AdbEnabled"),
        callback=lambda: self._toggle_param("AdbEnabled"),
      ),
      ssh_key_item("SSH Key", description=DESCRIPTIONS["ssh_key"]),
      toggle_item(
        "Joystick Debug Mode",
        description=DESCRIPTIONS["joystick_debug_mode"],
        initial_state=self._params.get_bool("JoystickDebugMode"),
        callback=lambda: self._toggle_param("JoystickDebugMode"),
      ),
      toggle_item(
        "Longitudinal Maneuver Mode",
        description="",
        initial_state=self._params.get_bool("LongitudinalManeuverMode"),
        callback=lambda: self._toggle_param("LongitudinalManeuverMode"),
      ),
      toggle_item(
        "openpilot Longitudinal Control (Alpha)",
        description="",
        initial_state=self._params.get_bool("AlphaLongitudinalEnabled"),
        callback=lambda: self._toggle_param("AlphaLongitudinalEnabled"),
      ),
      toggle_item(
        "Camping Mode",
        description="Enable offroad Miracast receiver for screen mirroring from phones.",
        initial_state=self._params.get_bool("CampingMode"),
        callback=lambda: self._toggle_param("CampingMode"),
      ),
      toggle_item(
        "Do Not Engage",
        description="Prevent engagement; device will not switch to driving state until you turn this off.",
        initial_state=self._params.get_bool("DoNotEngage"),
        callback=lambda: self._toggle_param("DoNotEngage"),
      ),
    ]

    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _render(self, rect):
    self._scroller.render(rect)

  def _toggle_param(self, key: str):
    current = self._params.get_bool(key)
    self._params.put_bool(key, not current)
