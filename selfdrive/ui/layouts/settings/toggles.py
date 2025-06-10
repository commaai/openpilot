from openpilot.system.ui.lib.application import Widget
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


class TogglesLayout(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()

    items = self._initialize_items()
    self._list_widget = ListView(items)

  def _initialize_items(self):
    def _param_toggle(title, param_name, icon):
      return toggle_item(
        title,
        DESCRIPTIONS.get(param_name, ""),
        icon=icon,
        initial_state=self._params.get_bool(param_name),
        callback=lambda state: self._params.put_bool(param_name, state),
      )

    items = [
      _param_toggle("Enable openpilot", "OpenpilotEnabledToggle", "chffr_wheel.png"),
      toggle_item("Experimental Mode", "", icon="experimental_white.png"),
      _param_toggle("Disengage on Accelerator Pedal", "DisengageOnAccelerator", "disengage_on_accelerator.png"),
      _param_toggle("Enable Lane Departure Warnings", "IsLdwEnabled", "warning.png"),
      _param_toggle("Always-On Driver Monitoring", "AlwaysOnDM", "monitoring.png"),
      _param_toggle("Record and Upload Driver Camera", "RecordFront", "monitoring.png"),
      _param_toggle("Use Metric System", "IsMetric", "monitoring.png"),
    ]
    return items

  def _render(self, rect):
    self._list_widget.render(rect)
