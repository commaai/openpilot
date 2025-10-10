from openpilot.common.params import Params
from openpilot.selfdrive.ui.widgets.ssh_key import ssh_key_item
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.list_view import toggle_item
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import DialogResult

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
    "<b>WARNING: openpilot longitudinal control is in alpha for this car and will disable Automatic Emergency Braking (AEB).</b><br><br>" +
    "On this car, openpilot defaults to the car's built-in ACC instead of openpilot's longitudinal control. " +
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
      callback=self._on_enable_adb,
    )

    # SSH enable toggle + SSH key management
    self._ssh_toggle = toggle_item(
      "Enable SSH",
      description="",
      initial_state=self._params.get_bool("SshEnabled"),
      callback=self._on_enable_ssh,
    )
    self._ssh_keys = ssh_key_item("SSH Keys", description=DESCRIPTIONS["ssh_key"])

    self._joystick_toggle = toggle_item(
      "Joystick Debug Mode",
      description="",
      initial_state=self._params.get_bool("JoystickDebugMode"),
      callback=self._on_joystick_debug_mode,
    )

    self._long_maneuver_toggle = toggle_item(
      "Longitudinal Maneuver Mode",
      description="",
      initial_state=self._params.get_bool("LongitudinalManeuverMode"),
      callback=self._on_long_maneuver_mode,
    )

    self._alpha_long_toggle = toggle_item(
      "openpilot Longitudinal Control (Alpha)",
      description=DESCRIPTIONS["alpha_longitudinal"],
      initial_state=self._params.get_bool("AlphaLongitudinalEnabled"),
      callback=self._on_alpha_long_enabled,
    )

    self._alpha_long_toggle.set_description(self._alpha_long_toggle.description + " Changing this setting will restart openpilot if the car is powered on.")

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
    # Hide non-release toggles on release builds
    for item in (self._adb_toggle, self._joystick_toggle, self._long_maneuver_toggle, self._alpha_long_toggle):
      item.set_visible(not self._is_release)

    # CP gating
    if ui_state.CP is not None:
      alpha_avail = True  # ui_state.CP.alphaLongitudinalAvailable
      if not alpha_avail or self._is_release:
        self._alpha_long_toggle.set_visible(False)
        self._params.remove("AlphaLongitudinalEnabled")
      else:
        self._alpha_long_toggle.set_visible(True)

      self._long_maneuver_toggle.action_item.set_enabled(ui_state.has_longitudinal_control and ui_state.is_offroad)
    else:
      self._long_maneuver_toggle.action_item.set_enabled(False)
      self._alpha_long_toggle.set_visible(False)

    # TODO: make a param control list item so we don't need to manage internal state as much here
    # refresh toggles from params to mirror external changes
    for key, item in (
      ("AdbEnabled", self._adb_toggle),
      ("SshEnabled", self._ssh_toggle),
      ("JoystickDebugMode", self._joystick_toggle),
      ("LongitudinalManeuverMode", self._long_maneuver_toggle),
      ("AlphaLongitudinalEnabled", self._alpha_long_toggle),
    ):
      item.action_item.set_state(self._params.get_bool(key))

  def _update_state(self):
    # Disable toggles that require onroad restart
    # TODO: we can do an onroad cycle, but alpha long toggle requires a deinit function to re-enable radar and not fault
    for item in (self._adb_toggle, self._joystick_toggle, self._long_maneuver_toggle, self._alpha_long_toggle):
      item.action_item.set_enabled(ui_state.is_offroad)

  def _on_enable_adb(self, state: bool):
    self._params.put_bool("AdbEnabled", state)

  def _on_enable_ssh(self, state: bool):
    self._params.put_bool("SshEnabled", state)

  def _on_joystick_debug_mode(self, state: bool):
    self._params.put_bool("JoystickDebugMode", state)
    self._params.put_bool("LongitudinalManeuverMode", False)
    self._long_maneuver_toggle.action_item.set_state(False)

  def _on_long_maneuver_mode(self, state: bool):
    self._params.put_bool("LongitudinalManeuverMode", state)
    self._params.put_bool("JoystickDebugMode", False)
    self._joystick_toggle.action_item.set_state(False)

  def _on_alpha_long_enabled(self, state: bool):
    if state:
      def confirm_callback(result: int):
        print('got result', result)
        if result == DialogResult.CONFIRM:
          self._params.put_bool("AlphaLongitudinalEnabled", True)
        else:
          self._alpha_long_toggle.action_item.set_state(False)

      # confirmation with desc
      content = (f"<h2 style=\"text-align: center;\">{self._alpha_long_toggle.title}</h2><br>"
                 f"<p style=\"text-align: center; font-size: 50px\">{self._alpha_long_toggle.description}</p>")

      dlg = ConfirmDialog(content, "Enable", rich=True)
      gui_app.set_modal_overlay(dlg, callback=confirm_callback)

    self._params.put_bool("AlphaLongitudinalEnabled", state)
    self._update_toggles()
