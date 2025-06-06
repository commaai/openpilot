from openpilot.system.ui.lib.application import Widget
from openpilot.system.ui.lib.list_view import ListView, text_item, button_item
from openpilot.common.params import Params
from openpilot.system.hardware import TICI
from openpilot.system.ui.widgets.confirm_dialog import confirm_dialog, alert_dialog
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.ui_state import ui_state

# Description constants
DESCRIPTIONS = {
  'pair_device': "Pair your device with comma connect (connect.comma.ai) and claim your comma prime offer.",
  'driver_camera': "Preview the driver facing camera to ensure that driver monitoring has good visibility. (vehicle must be off)",
  'reset_calibration': (
    "openpilot requires the device to be mounted within 4° left or right and within 5° " +
    "up or 9° down. openpilot is continuously calibrating, resetting is rarely required."
  ),
  'review_guide': "Review the rules, features, and limitations of openpilot",
}


class DeviceLayout(Widget):
  def __init__(self):
    super().__init__()

    self.params = Params()
    dongle_id = self.params.get("DongleId", encoding="utf-8") or "N/A"
    serial = self.params.get("HardwareSerial") or "N/A"

    items = [
      text_item("Dongle ID", dongle_id),
      text_item("Serial", serial),
      button_item("Pair Device", "PAIR", DESCRIPTIONS['pair_device'], self._on_pair_device),
      button_item("Driver Camera", "PREVIEW", DESCRIPTIONS['driver_camera'], self._on_driver_camera),
      button_item("Reset Calibration", "RESET", DESCRIPTIONS['reset_calibration'], callback=self._on_reset_calibration),
      button_item("Review Training Guide", "REVIEW", DESCRIPTIONS['review_guide'], self._on_review_training_guide),
    ]

    if TICI:
      items.append(button_item("Regulatory", "VIEW", callback=self._on_regulatory))

    items.append(button_item("Change Language", "CHANGE", callback=self._on_change_language))

    self._list_widget = ListView(items)

  def _render(self, rect):
    self._list_widget.render(rect)


  def _on_reset_calibration(self):
    if ui_state.engaged:
      gui_app.set_modal_overlay(lambda: alert_dialog("Disengage to Reset Calibration"))
      return

    gui_app.set_modal_overlay(
      lambda: confirm_dialog("Are you sure you want to reset calibration?", "Reset"),
      callback=self._reset_calibration,
    )

  def _reset_calibration(self, result: int):
    if ui_state.engaged or result != 1:
      return

    self._params.remove("CalibrationParams")
    self._params.remove("LiveTorqueParameters")
    self._params.remove("LiveParameters")
    self._params.remove("LiveParametersV2")
    self._params.remove("LiveDelay")
    self._params.put_bool("OnroadCycleRequested", True)

  def _on_pair_device(self): pass
  def _on_driver_camera(self): pass
  def _on_review_training_guide(self): pass
  def _on_regulatory(self): pass
  def _on_change_language(self): pass
