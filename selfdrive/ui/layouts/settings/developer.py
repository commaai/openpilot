from openpilot.common.params import Params
from openpilot.selfdrive.ui.widgets.ssh_key import ssh_key_item
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.list_view import toggle_item
from openpilot.system.ui.widgets.scroller import Scroller

# Description constants
DESCRIPTIONS = {
  'enable_adb': (
    "ADB (Android Debug Bridge) allows connecting to your device over USB or over the network. " +
    "See https://docs.comma.ai/how-to/connect-to-comma for more info."
  ),
  'ssh_key': (
    "Warning: This grants SSH access to all public keys in your GitHub settings. Never enter a GitHub username " +
    "other than your own. A comma employee will NEVER ask you to add their GitHub username."
  ),
  'alpha_longitudinal': (
    "<b>WARNING: openpilot longitudinal control is in alpha for this car and will disable Automatic Emergency Braking (AEB).</b><br><br>"
    "On this car, openpilot defaults to the car's built-in ACC instead of openpilot's longitudinal control. "
    "Enable this to switch to openpilot longitudinal control. Enabling Experimental mode is recommended when enabling openpilot longitudinal control alpha."
  ),
}


class DeveloperLayout(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()
    self._is_release = self._params.get_bool("IsReleaseBranch")

    # Build items and keep references for callbacks/state updates
    self._adb_toggle = toggle_item(
      "Enable ADB",
      description=DESCRIPTIONS["enable_adb"],
      initial_state=self._params.get_bool("AdbEnabled"),
      callback=lambda state: self._on_enable_adb(state),
    )

    # SSH enable toggle + SSH key management
    self._ssh_toggle = toggle_item(
      "Enable SSH",
      description="",
      initial_state=self._params.get_bool("SshEnabled"),
      callback=lambda state: self._on_enable_ssh(state),
    )
    self._ssh_keys = ssh_key_item("SSH Keys", description=DESCRIPTIONS["ssh_key"])

    self._joystick_toggle = toggle_item(
      "Joystick Debug Mode",
      description="",
      initial_state=self._params.get_bool("JoystickDebugMode"),
      callback=lambda state: self._on_joystick_debug_mode(state),
    )

    self._long_maneuver_toggle = toggle_item(
      "Longitudinal Maneuver Mode",
      description="",
      initial_state=self._params.get_bool("LongitudinalManeuverMode"),
      callback=lambda state: self._on_long_maneuver_mode(state),
    )

    self._alpha_long_toggle = toggle_item(
      "openpilot Longitudinal Control (Alpha)",
      description=DESCRIPTIONS["alpha_longitudinal"],
      initial_state=self._params.get_bool("AlphaLongitudinalEnabled"),
      callback=lambda state: self._on_alpha_long_enabled(state),
    )

    items = [
      self._adb_toggle,
      self._ssh_toggle,
      self._ssh_keys,
      self._joystick_toggle,
      self._long_maneuver_toggle,
      self._alpha_long_toggle,
    ]

    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _render(self, rect):
    self._scroller.render(rect)

  def show_event(self):
    self._update_toggles()

  def _update_toggles(self):
    offroad = not ui_state.started

    # Hide Param-like toggles on release branches (SSH key mgmt remains)
    for item in (self._adb_toggle, self._joystick_toggle, self._long_maneuver_toggle, self._alpha_long_toggle):
      item.set_visible(not self._is_release)

    # Enable/disable on offroad (alpha toggle stays enabled; CP gating below may override)
    for item in (self._adb_toggle, self._joystick_toggle, self._long_maneuver_toggle):
      item.action_item.set_enabled(offroad)

    # CP gating
    if ui_state.CP is not None:
      alpha_avail = ui_state.CP.alphaLongitudinalAvailable
      if (not alpha_avail) or self._is_release:
        self._params.remove("AlphaLongitudinalEnabled")
        self._alpha_long_toggle.action_item.set_enabled(False)
      self._alpha_long_toggle.set_visible(alpha_avail and (not self._is_release))

      self._long_maneuver_toggle.action_item.set_enabled(ui_state.has_longitudinal_control and offroad)
    else:
      self._long_maneuver_toggle.action_item.set_enabled(False)
      self._alpha_long_toggle.set_visible(False)

    # Sync alpha toggle state from params (Qt refresh equivalent)
    self._alpha_long_toggle.action_item.set_state(self._params.get_bool("AlphaLongitudinalEnabled"))

  def _on_enable_adb(self, state: bool):
    self._params.put_bool("AdbEnabled", state)

  def _on_enable_ssh(self, state: bool):
    self._params.put_bool("SshEnabled", state)

  def _on_joystick_debug_mode(self, state: bool):
    self._params.put_bool("JoystickDebugMode", state)
    if state:
      self._params.put_bool("LongitudinalManeuverMode", False)
      self._long_maneuver_toggle.action_item.set_state(False)

  def _on_long_maneuver_mode(self, state: bool):
    self._params.put_bool("LongitudinalManeuverMode", state)
    if state:
      self._params.put_bool("JoystickDebugMode", False)
      self._joystick_toggle.action_item.set_state(False)

  def _on_alpha_long_enabled(self, state: bool):
    self._params.put_bool("AlphaLongitudinalEnabled", state)
    self._update_toggles()
