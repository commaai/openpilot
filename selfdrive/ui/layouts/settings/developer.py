from openpilot.system.ui.lib.list_view import ListView,  toggle_item
from openpilot.common.params import Params

# Description constants
DESCRIPTIONS = {
  'enable_adb': (
    "ADB (Android Debug Bridge) allows connecting to your device over USB or over the network. " +
    "See https://docs.comma.ai/how-to/connect-to-comma for more info."
  ),
  'joystick_debug_mode': "Preview the driver facing camera to ensure that driver monitoring has good visibility. (vehicle must be off)",
}


class DeveloperLayout:
  def __init__(self):
    self._params = Params()
    items = [
      toggle_item(
        "Enable ADB",
        description=DESCRIPTIONS["enable_adb"],
        initial_state=self._params.get_bool("AdbEnabled"),
        callback=self._on_enable_adb,
      ),
      toggle_item(
        "Joystick Debug Mode",
        description=DESCRIPTIONS["joystick_debug_mode"],
        initial_state=self._params.get_bool("JoystickDebugMode"),
        callback=self._on_joystick_debug_mode,
      ),
      toggle_item(
        "Longitudinal Maneuver Mode",
        description="",
        initial_state=self._params.get_bool("LongitudinalManeuverMode"),
        callback=self._on_long_maneuver_mode,
      ),
      toggle_item(
        "openpilot Longitudinal Control (Alpha)",
        description="",
        initial_state=self._params.get_bool("AlphaLongitudinalEnabled"),
        callback=self._on_alpha_long_enabled,
      ),
    ]

    self._list_widget = ListView(items)

  def render(self, rect):
    self._list_widget.render(rect)

  def _on_enable_adb(self): pass
  def _on_joystick_debug_mode(self): pass
  def _on_long_maneuver_mode(self): pass
  def _on_alpha_long_enabled(self): pass
