from openpilot.common.params import Params, UnknownKeyName
from openpilot.system.ui.lib.list_view import ListView, MultipleButtonItem, ToggleItem
from openpilot.system.ui.lib.widget import Widget
from openpilot.selfdrive.ui.ui_state import ui_state


# Description constants
DESCRIPTIONS = {
  "OpenpilotEnabledToggle": (
    "Use the openpilot system for adaptive cruise control and lane keep driver assistance. " +
    "Your attention is required at all times to use this feature."
  ),
  "DisengageOnAccelerator": "When enabled, pressing the accelerator pedal will disengage openpilot.",
  "LongitudinalPersonality": (
    "Standard is recommended. In aggressive mode, openpilot will follow lead cars closer and be more aggressive with the gas and brake. " +
    "In relaxed mode openpilot will stay further away from lead cars. On supported cars, you can cycle through these personalities with " +
    "your steering wheel distance button."
  ),
  "IsLdwEnabled": (
    "Receive alerts to steer back into the lane when your vehicle drifts over a detected lane line " +
    "without a turn signal activated while driving over 31 mph (50 km/h)."
  ),
  "AlwaysOnDM": "Enable driver monitoring even when openpilot is not engaged.",
  'RecordFront': "Upload data from the driver facing camera and help improve the driver monitoring algorithm.",
  "IsMetric": "Display speed in km/h instead of mph.",
}


class ParamsToggleItem(ToggleItem):
  def __init__(self, param_name, title, description, init_state, locked=False, need_restart=False, **kwargs):
    super().__init__(title, description, init_state, **kwargs)
    self.param_name = param_name
    self.locked = locked
    self.need_restart = need_restart


class TogglesLayout(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()

    self.personality_item = MultipleButtonItem(
      "Driving Personality",
      DESCRIPTIONS["LongitudinalPersonality"],
      buttons=["Aggressive", "Standard", "Relaxed"],
      button_width=255,
      callback=self._set_longitudinal_personality,
      selected_index=int(self._params.get("LongitudinalPersonality") or 0),
      icon="speed_limit.png",
    )

    items = [
      self._param_toggle("Enable openpilot", "OpenpilotEnabledToggle", "chffr_wheel.png", need_restart=True),
      self._param_toggle("Experimental Mode", "ExperimentalMode", "experimental_white.png", active_icon="experimental.png"),
      self._param_toggle("Disengage on Accelerator Pedal", "DisengageOnAccelerator", "disengage_on_accelerator.png"),
      self.personality_item,
      self._param_toggle("Enable Lane Departure Warnings", "IsLdwEnabled", "warning.png"),
      self._param_toggle("Always-On Driver Monitoring", "AlwaysOnDM", "monitoring.png"),
      self._param_toggle("Record and Upload Driver Camera", "RecordFront", "monitoring.png", need_restart=True),
      self._param_toggle("Use Metric System", "IsMetric", "monitoring.png"),
    ]
    self._list_widget = ListView(items)

  def _param_toggle(self, title, param_name, icon, need_restart=False, active_icon=None):
    try:
      locked = self._params.get_bool(f"{param_name}Lock")
    except UnknownKeyName:
      locked = False

    desc = DESCRIPTIONS.get(param_name, "")
    if need_restart and not locked:
      desc += " Changing this setting will restart openpilot if the car is powered on."

    return ParamsToggleItem(
      param_name,  title, desc, self._params.get_bool(param_name), icon=icon,
      need_restart=need_restart,
      callback=self._on_toggle_change,
      active_icon=active_icon,
      enabled=lambda: (not locked) and (not need_restart or ui_state.is_offroad())
     )

  def _render(self, rect):
    # Update personality item from selfdriveState if available
    if ui_state.sm.updated["selfdriveState"]:
      self.personality_item.selected_index = ui_state.sm["selfdriveState"].personality

    self._list_widget.render(rect)

  def _set_longitudinal_personality(self, button_index: int):
    self._params.put("LongitudinalPersonality", str(button_index))

  def _on_toggle_change(self, item: ParamsToggleItem):
    self._params.put_bool(item.param_name, item.get_state())
    if item.need_restart:
      self._params.put_bool("OnroadCycleRequested", True)
