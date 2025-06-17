from openpilot.system.ui.lib.list_view import ListView, toggle_item
from openpilot.common.params import Params

# Description constants
DESCRIPTIONS = {
  "OpenpilotEnabledToggle": (
    "Use the openpilot system for adaptive cruise control and lane keep driver assistance. " +
    "Your attention is required at all times to use this feature."
  ),
  "DisengageOnAccelerator": "When enabled, pressing the accelerator pedal will disengage openpilot.",
  "IsLdwEnabled": (
    "Receive alerts to steer back into the lane when your vehicle drifts over a detected lane line " +
     "without a turn signal activated while driving over 31 mph (50 km/h)."
  ),
  "AlwaysOnDM": "Enable driver monitoring even when openpilot is not engaged.",
  'RecordFront': "Upload data from the driver facing camera and help improve the driver monitoring algorithm.",
  "IsMetric": "Display speed in km/h instead of mph.",
}


class TogglesLayout:
  def __init__(self):
    self._params = Params()
    items = [
      toggle_item(
        "Enable openpilot",
        DESCRIPTIONS["OpenpilotEnabledToggle"],
        self._params.get_bool("OpenpilotEnabledToggle"),
        icon="chffr_wheel.png",
      ),
      toggle_item(
        "Experimental Mode",
        initial_state=self._params.get_bool("ExperimentalMode"),
        icon="experimental_white.png",
      ),
      toggle_item(
        "Disengage on Accelerator Pedal",
        DESCRIPTIONS["DisengageOnAccelerator"],
        self._params.get_bool("DisengageOnAccelerator"),
        icon="disengage_on_accelerator.png",
      ),
      toggle_item(
        "Enable Lane Departure Warnings",
        DESCRIPTIONS["IsLdwEnabled"],
        self._params.get_bool("IsLdwEnabled"),
        icon="warning.png",
      ),
      toggle_item(
        "Always-On Driver Monitoring",
        DESCRIPTIONS["AlwaysOnDM"],
        self._params.get_bool("AlwaysOnDM"),
        icon="monitoring.png",
      ),
      toggle_item(
        "Record and Upload Driver Camera",
        DESCRIPTIONS["RecordFront"],
        self._params.get_bool("RecordFront"),
        icon="monitoring.png",
      ),
      toggle_item(
        "Use Metric System", DESCRIPTIONS["IsMetric"], self._params.get_bool("IsMetric"), icon="monitoring.png"
      ),
    ]

    self._list_widget = ListView(items)

  def render(self, rect):
    self._list_widget.render(rect)
