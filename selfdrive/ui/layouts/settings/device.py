import os
import json

from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.selfdrive.ui.onroad.driver_camera_dialog import DriverCameraDialog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.widgets.pairing_dialog import PairingDialog
from openpilot.system.hardware import TICI
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.confirm_dialog import confirm_dialog, alert_dialog
from openpilot.system.ui.widgets.html_render import HtmlRenderer
from openpilot.system.ui.widgets.list_view import text_item, button_item, dual_button_item
from openpilot.system.ui.widgets.option_dialog import MultiOptionDialog
from openpilot.system.ui.widgets.scroller import Scroller

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
    self._select_language_dialog: MultiOptionDialog | None = None
    self._driver_camera: DriverCameraDialog | None = None
    self._pair_device_dialog: PairingDialog | None = None
    self._fcc_dialog: HtmlRenderer | None = None

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    dongle_id = self._params.get("DongleId", encoding="utf-8") or "N/A"
    serial = self._params.get("HardwareSerial") or "N/A"

    items = [
      text_item("Dongle ID", dongle_id),
      text_item("Serial", serial),
      button_item("Pair Device", "PAIR", DESCRIPTIONS['pair_device'], callback=self._pair_device),
      button_item("Driver Camera", "PREVIEW", DESCRIPTIONS['driver_camera'], callback=self._show_driver_camera, enabled=ui_state.is_offroad),
      button_item("Reset Calibration", "RESET", DESCRIPTIONS['reset_calibration'], callback=self._reset_calibration_prompt),
      regulatory_btn := button_item("Regulatory", "VIEW", callback=self._on_regulatory),
      button_item("Review Training Guide", "REVIEW", DESCRIPTIONS['review_guide'], self._on_review_training_guide),
      button_item("Change Language", "CHANGE", callback=self._show_language_selection, enabled=ui_state.is_offroad),
      dual_button_item("Reboot", "Power Off", left_callback=self._reboot_prompt, right_callback=self._power_off_prompt),
    ]
    regulatory_btn.set_visible(TICI)
    return items

  def _render(self, rect):
    self._scroller.render(rect)

  def _show_language_selection(self):
    try:
      languages_file = os.path.join(BASEDIR, "selfdrive/ui/translations/languages.json")
      with open(languages_file, encoding='utf-8') as f:
        languages = json.load(f)

      self._select_language_dialog = MultiOptionDialog("Select a language", languages)
      gui_app.set_modal_overlay(self._select_language_dialog, callback=self._handle_language_selection)
    except FileNotFoundError:
      pass

  def _handle_language_selection(self, result: int):
    if result == 1 and self._select_language_dialog:
      selected_language = self._select_language_dialog.selection
      self._params.put("LanguageSetting", selected_language)

    self._select_language_dialog = None

  def _show_driver_camera(self):
    if not self._driver_camera:
      self._driver_camera = DriverCameraDialog()

    gui_app.set_modal_overlay(self._driver_camera, callback=lambda result: setattr(self, '_driver_camera', None))

  def _reset_calibration_prompt(self):
    if ui_state.engaged:
      gui_app.set_modal_overlay(lambda: alert_dialog("Disengage to Reset Calibration"))
      return

    gui_app.set_modal_overlay(
      lambda: confirm_dialog("Are you sure you want to reset calibration?", "Reset"),
      callback=self._reset_calibration,
    )

  def _reset_calibration(self, result: int):
    if ui_state.engaged or result != DialogResult.CONFIRM:
      return

    self._params.remove("CalibrationParams")
    self._params.remove("LiveTorqueParameters")
    self._params.remove("LiveParameters")
    self._params.remove("LiveParametersV2")
    self._params.remove("LiveDelay")
    self._params.put_bool("OnroadCycleRequested", True)

  def _reboot_prompt(self):
    if ui_state.engaged:
      gui_app.set_modal_overlay(lambda: alert_dialog("Disengage to Reboot"))
      return

    gui_app.set_modal_overlay(
      lambda: confirm_dialog("Are you sure you want to reboot?", "Reboot"),
      callback=self._perform_reboot,
    )

  def _perform_reboot(self, result: int):
    if not ui_state.engaged and result == DialogResult.CONFIRM:
      self._params.put_bool_nonblocking("DoReboot", True)

  def _power_off_prompt(self):
    if ui_state.engaged:
      gui_app.set_modal_overlay(lambda: alert_dialog("Disengage to Power Off"))
      return

    gui_app.set_modal_overlay(
      lambda: confirm_dialog("Are you sure you want to power off?", "Power Off"),
      callback=self._perform_power_off,
    )

  def _perform_power_off(self, result: int):
    if not ui_state.engaged and result == DialogResult.CONFIRM:
      self._params.put_bool_nonblocking("DoShutdown", True)

  def _pair_device(self):
    if not self._pair_device_dialog:
      self._pair_device_dialog = PairingDialog()
    gui_app.set_modal_overlay(self._pair_device_dialog, callback=lambda result: setattr(self, '_pair_device_dialog', None))

  def _on_regulatory(self):
    if not self._fcc_dialog:
      self._fcc_dialog = HtmlRenderer(os.path.join(BASEDIR, "selfdrive/assets/offroad/fcc.html"))

    gui_app.set_modal_overlay(self._fcc_dialog,
      callback=lambda result: setattr(self, '_fcc_dialog', None),
    )

  def _on_review_training_guide(self): pass
