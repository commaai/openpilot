from openpilot.common.params import Params
from openpilot.selfdrive.ui.widgets.ssh_key import ssh_key_item
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.list_view import ParamToggleAction, param_toggle_item, toggle_item
from openpilot.system.ui.widgets.scroller_tici import Scroller
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.widgets import DialogResult

# Description constants
DESCRIPTIONS = {
  'enable_adb': tr_noop(
    "ADB (Android Debug Bridge) allows connecting to your device over USB or over the network. " +
    "See https://docs.comma.ai/how-to/connect-to-comma for more info."
  ),
  'ssh_key': tr_noop(
    "Warning: This grants SSH access to all public keys in your GitHub settings. Never enter a GitHub username " +
    "other than your own. A comma employee will NEVER ask you to add their GitHub username."
  ),
  'alpha_longitudinal': tr_noop(
    "<b>WARNING: openpilot longitudinal control is in alpha for this car and may disable Automatic Emergency Braking (AEB).</b><br><br>" +
    "On this car, openpilot defaults to the car's built-in ACC instead of openpilot's longitudinal control. " +
    "Enable this to switch to openpilot longitudinal control. Enabling Experimental mode is recommended when enabling openpilot longitudinal control alpha. " +
    "Changing this setting will restart openpilot if the car is powered on."
  ),
}


class DeveloperLayout(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()
    self._is_release = self._params.get_bool("IsReleaseBranch")

    # Build items and keep references for callbacks/state updates
    self._adb_toggle = param_toggle_item(
      "AdbEnabled",
      lambda: tr("Enable ADB"),
      description=lambda: tr(DESCRIPTIONS["enable_adb"]),
      enabled=ui_state.is_offroad,
    )

    # SSH enable toggle + SSH key management
    self._ssh_toggle = param_toggle_item("SshEnabled", lambda: tr("Enable SSH"), description="")
    self._ssh_keys = ssh_key_item(lambda: tr("SSH Keys"), description=lambda: tr(DESCRIPTIONS["ssh_key"]))

    self._joystick_toggle = param_toggle_item(
      "JoystickDebugMode",
      lambda: tr("Joystick Debug Mode"),
      description="",
      callback=self._on_joystick_debug_mode,
      enabled=ui_state.is_offroad,
    )

    self._long_maneuver_toggle = param_toggle_item(
      "LongitudinalManeuverMode",
      lambda: tr("Longitudinal Maneuver Mode"),
      description="",
      callback=self._on_long_maneuver_mode,
    )

    self._lat_maneuver_toggle = param_toggle_item(
      "LateralManeuverMode",
      lambda: tr("Lateral Maneuver Mode"),
      description="",
      callback=self._on_lat_maneuver_mode,
    )

    # alpha long writes its param only after confirmation, so it isn't a param toggle
    self._alpha_long_toggle = toggle_item(
      lambda: tr("openpilot Longitudinal Control (Alpha)"),
      description=lambda: tr(DESCRIPTIONS["alpha_longitudinal"]),
      initial_state=self._params.get_bool("AlphaLongitudinalEnabled"),
      callback=self._on_alpha_long_enabled,
      enabled=lambda: not ui_state.engaged,
    )

    self._ui_debug_toggle = param_toggle_item(
      "ShowDebugInfo",
      lambda: tr("UI Debug Mode"),
      description="",
      callback=self._on_enable_ui_debug,
    )
    self._on_enable_ui_debug(self._ui_debug_toggle.action_item.get_state())

    self._items = [
      self._adb_toggle,
      self._ssh_toggle,
      self._ssh_keys,
      self._joystick_toggle,
      self._long_maneuver_toggle,
      self._lat_maneuver_toggle,
      self._alpha_long_toggle,
      self._ui_debug_toggle,
    ]
    self._scroller = Scroller(self._items, line_separator=True, spacing=0)

    # Toggles should be not available to change in onroad state
    ui_state.add_offroad_transition_callback(self._update_toggles)

  def _render(self, rect):
    self._scroller.render(rect)

  def show_event(self):
    super().show_event()
    self._scroller.show_event()
    self._update_toggles()

  def _update_toggles(self):
    ui_state.update_params()

    # Hide non-release toggles on release builds
    # TODO: we can do an onroad cycle, but alpha long toggle requires a deinit function to re-enable radar and not fault
    for item in (self._joystick_toggle, self._long_maneuver_toggle, self._lat_maneuver_toggle, self._alpha_long_toggle):
      item.set_visible(not self._is_release)

    # CP gating
    if ui_state.CP is not None:
      alpha_avail = ui_state.CP.alphaLongitudinalAvailable
      if not alpha_avail or self._is_release:
        self._alpha_long_toggle.set_visible(False)
        self._params.remove("AlphaLongitudinalEnabled")
      else:
        self._alpha_long_toggle.set_visible(True)

      long_man_enabled = ui_state.has_longitudinal_control and ui_state.is_offroad()
      self._long_maneuver_toggle.action_item.set_enabled(long_man_enabled)
      self._lat_maneuver_toggle.action_item.set_enabled(ui_state.is_offroad())
    else:
      self._long_maneuver_toggle.action_item.set_enabled(False)
      self._lat_maneuver_toggle.action_item.set_enabled(False)
      self._alpha_long_toggle.set_visible(False)

    # refresh toggles from params to mirror external changes
    for item in self._items:
      if isinstance(item.action_item, ParamToggleAction):
        item.action_item.refresh()
    self._alpha_long_toggle.action_item.set_state(self._params.get_bool("AlphaLongitudinalEnabled"))

  def _on_enable_ui_debug(self, state: bool):
    gui_app.set_show_touches(state)
    gui_app.set_show_fps(state)

  def _clear_modes(self, *items):
    # the debug/maneuver modes are mutually exclusive
    for item in items:
      self._params.put_bool(item.action_item.param, False, block=True)
      item.action_item.refresh()

  def _on_joystick_debug_mode(self, state: bool):
    self._clear_modes(self._long_maneuver_toggle, self._lat_maneuver_toggle)

  def _on_long_maneuver_mode(self, state: bool):
    self._clear_modes(self._joystick_toggle, self._lat_maneuver_toggle)

  def _on_lat_maneuver_mode(self, state: bool):
    self._params.put_bool("ExperimentalMode", False, block=True)
    self._clear_modes(self._joystick_toggle, self._long_maneuver_toggle)

  def _on_alpha_long_enabled(self, state: bool):
    if state:
      def confirm_callback(result: DialogResult):
        if result == DialogResult.CONFIRM:
          self._params.put_bool("AlphaLongitudinalEnabled", True, block=True)
          self._params.put_bool("OnroadCycleRequested", True, block=True)
          self._update_toggles()
        else:
          self._alpha_long_toggle.action_item.set_state(False)

      # show confirmation dialog
      content = (f"<h1>{self._alpha_long_toggle.title}</h1><br>" +
                 f"<p>{self._alpha_long_toggle.description}</p>")

      dlg = ConfirmDialog(content, tr("Enable"), rich=True, callback=confirm_callback)
      gui_app.push_widget(dlg)

    else:
      self._params.put_bool("AlphaLongitudinalEnabled", False, block=True)
      self._params.put_bool("OnroadCycleRequested", True, block=True)
      self._update_toggles()
