from openpilot.system.ui.lib.list_view import ListView, MultipleButtonItem, ToggleItem
from openpilot.system.ui.lib.widget import Widget
from openpilot.common.params import Params

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


class TogglesLayout(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()
    items = [
      ToggleItem(
        "Enable openpilot",
        DESCRIPTIONS["OpenpilotEnabledToggle"],
        self._params.get_bool("OpenpilotEnabledToggle"),
        icon="chffr_wheel.png",
      ),
      ToggleItem(
        "Experimental Mode",
        initial_state=self._params.get_bool("ExperimentalMode"),
        icon="experimental_white.png",
      ),
      ToggleItem(
        "Disengage on Accelerator Pedal",
        DESCRIPTIONS["DisengageOnAccelerator"],
        self._params.get_bool("DisengageOnAccelerator"),
        icon="disengage_on_accelerator.png",
      ),
      MultipleButtonItem(
        "Driving Personality",
        DESCRIPTIONS["LongitudinalPersonality"],
        buttons=["Aggressive", "Standard", "Relaxed"],
        button_width=255,
        callback=self._set_longitudinal_personality,
        selected_index=int(self._params.get("LongitudinalPersonality") or 0),
        icon="speed_limit.png"
      ),
      ToggleItem(
        "Enable Lane Departure Warnings",
        DESCRIPTIONS["IsLdwEnabled"],
        self._params.get_bool("IsLdwEnabled"),
        icon="warning.png",
      ),
      ToggleItem(
        "Always-On Driver Monitoring",
        DESCRIPTIONS["AlwaysOnDM"],
        self._params.get_bool("AlwaysOnDM"),
        icon="monitoring.png",
      ),
      ToggleItem(
        "Record and Upload Driver Camera",
        DESCRIPTIONS["RecordFront"],
        self._params.get_bool("RecordFront"),
        icon="monitoring.png",
      ),
      ToggleItem(
        "Use Metric System", DESCRIPTIONS["IsMetric"], self._params.get_bool("IsMetric"), icon="monitoring.png"
      ),
    ]

    self._list_widget = ListView(items)

  def _render(self, rect):
    self._list_widget.render(rect)

  def _set_longitudinal_personality(self, button_index: int):
    self._params.put("LongitudinalPersonality", str(button_index))
