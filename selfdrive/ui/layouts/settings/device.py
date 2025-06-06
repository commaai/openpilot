import os
import json
from openpilot.system.ui.lib.application import gui_app, Widget
from openpilot.system.ui.lib.list_view import ListView, text_item, button_item
from openpilot.selfdrive.ui.onroad.driver_camera_view import DriverCameraView
from openpilot.common.params import Params
from openpilot.system.ui.widgets.option_dialog import MultiOptionDialog
from openpilot.system.hardware import TICI
from openpilot.common.basedir import BASEDIR


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

    self._params = Params()
    dongle_id = self._params.get("DongleId", encoding="utf-8") or "N/A"
    serial = self._params.get("HardwareSerial") or "N/A"

    items = [
      text_item("Dongle ID", dongle_id),
      text_item("Serial", serial),
      button_item("Pair Device", "PAIR", DESCRIPTIONS['pair_device'], self._on_pair_device),
      button_item("Driver Camera", "PREVIEW", DESCRIPTIONS['driver_camera'], callback=self._on_driver_camera),
      button_item("Reset Calibration", "RESET", DESCRIPTIONS['reset_calibration'], self._on_reset_calibration),
      button_item("Review Training Guide", "REVIEW", DESCRIPTIONS['review_guide'], self._on_review_training_guide),
    ]

    if TICI:
      items.append(button_item("Regulatory", "VIEW", callback=self._on_regulatory))

    items.append(button_item("Change Language", "CHANGE", callback=self._on_change_language))

    self._list_widget = ListView(items)
    self._select_language_dialog: MultiOptionDialog | None = None
    self._driver_camera = DriverCameraView()

  def _render(self, rect):
    self._list_widget.render(rect)

  def _on_change_language(self):
    try:
      languages_file = os.path.join(BASEDIR, "selfdrive/ui/translations/languages.json")
      with open(languages_file, encoding='utf-8') as f:
        languages = json.load(f)

      self._select_language_dialog = MultiOptionDialog("Select a language", languages)
      gui_app.set_modal_overlay(self._select_language_dialog, callback=self._on_select_lang_dialog_closed)
    except FileNotFoundError:
      pass

  def _on_select_lang_dialog_closed(self, result: int):
    if result == 1 and self._select_language_dialog:
      selected_language = self._select_language_dialog.selection
      self._params.put("LanguageSetting", selected_language)

    self._select_language_dialog = None

  def _on_driver_camera(self):
    if not self._driver_camera:
      self._driver_camera = DriverCameraView()

    gui_app.set_modal_overlay(self._driver_camera, callback=lambda result: setattr(self, '_driver_camera', None))

  def _on_pair_device(self): pass
  def _on_reset_calibration(self): pass
  def _on_review_training_guide(self): pass
  def _on_regulatory(self): pass
