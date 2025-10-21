import os
import json
import math

from cereal import messaging, log
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.ui.onroad.driver_camera_dialog import DriverCameraDialog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.layouts.onboarding import TrainingGuide
from openpilot.selfdrive.ui.widgets.pairing_dialog import PairingDialog
from openpilot.system.hardware import TICI
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog, alert_dialog
from openpilot.system.ui.widgets.html_render import HtmlModal
from openpilot.system.ui.widgets.list_view import text_item, button_item, dual_button_item
from openpilot.system.ui.widgets.option_dialog import MultiOptionDialog
from openpilot.system.ui.widgets.scroller import Scroller

# Description constants
DESCRIPTIONS = {
  'pair_device': tr("Pair your device with comma connect (connect.comma.ai) and claim your comma prime offer."),
  'driver_camera': tr("Preview the driver facing camera to ensure that driver monitoring has good visibility. (vehicle must be off)"),
  'reset_calibration': tr("openpilot requires the device to be mounted within 4° left or right and within 5° up or 9° down."),
  'review_guide': tr("Review the rules, features, and limitations of openpilot"),
}


class DeviceLayout(Widget):
  def __init__(self):
    super().__init__()

    self._params = Params()
    self._select_language_dialog: MultiOptionDialog | None = None
    self._driver_camera: DriverCameraDialog | None = None
    self._pair_device_dialog: PairingDialog | None = None
    self._fcc_dialog: HtmlModal | None = None
    self._training_guide: TrainingGuide | None = None

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

    ui_state.add_offroad_transition_callback(self._offroad_transition)

  def _initialize_items(self):
    dongle_id = self._params.get("DongleId") or tr("N/A")
    serial = self._params.get("HardwareSerial") or tr("N/A")

    self._pair_device_btn = button_item(tr("Pair Device"), tr("PAIR"), DESCRIPTIONS['pair_device'], callback=self._pair_device)
    self._pair_device_btn.set_visible(lambda: not ui_state.prime_state.is_paired())

    self._reset_calib_btn = button_item(tr("Reset Calibration"), tr("RESET"), DESCRIPTIONS['reset_calibration'], callback=self._reset_calibration_prompt)
    self._reset_calib_btn.set_description_opened_callback(self._update_calib_description)

    self._power_off_btn = dual_button_item(tr("Reboot"), tr("Power Off"), left_callback=self._reboot_prompt, right_callback=self._power_off_prompt)

    items = [
      text_item(tr("Dongle ID"), dongle_id),
      text_item(tr("Serial"), serial),
      self._pair_device_btn,
      button_item(tr("Driver Camera"), tr("PREVIEW"), DESCRIPTIONS['driver_camera'], callback=self._show_driver_camera, enabled=ui_state.is_offroad),
      self._reset_calib_btn,
      button_item(tr("Review Training Guide"), tr("REVIEW"), DESCRIPTIONS['review_guide'], self._on_review_training_guide, enabled=ui_state.is_offroad),
      regulatory_btn := button_item(tr("Regulatory"), tr("VIEW"), callback=self._on_regulatory, enabled=ui_state.is_offroad),
      # TODO: implement multilang
      # button_item(tr("Change Language"), tr("CHANGE"), callback=self._show_language_selection, enabled=ui_state.is_offroad),
      self._power_off_btn,
    ]
    regulatory_btn.set_visible(TICI)
    return items

  def _offroad_transition(self):
    self._power_off_btn.action_item.right_button.set_visible(ui_state.is_offroad())

  def show_event(self):
    self._scroller.show_event()

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
      gui_app.set_modal_overlay(alert_dialog(tr("Disengage to Reset Calibration")))
      return

    def reset_calibration(result: int):
      # Check engaged again in case it changed while the dialog was open
      if ui_state.engaged or result != DialogResult.CONFIRM:
        return

      self._params.remove("CalibrationParams")
      self._params.remove("LiveTorqueParameters")
      self._params.remove("LiveParameters")
      self._params.remove("LiveParametersV2")
      self._params.remove("LiveDelay")
      self._params.put_bool("OnroadCycleRequested", True)
      self._update_calib_description()

    dialog = ConfirmDialog(tr("Are you sure you want to reset calibration?"), tr("Reset"))
    gui_app.set_modal_overlay(dialog, callback=reset_calibration)

  def _update_calib_description(self):
    desc = DESCRIPTIONS['reset_calibration']

    calib_bytes = self._params.get("CalibrationParams")
    if calib_bytes:
      try:
        calib = messaging.log_from_bytes(calib_bytes, log.Event).liveCalibration

        if calib.calStatus != log.LiveCalibrationData.Status.uncalibrated:
          pitch = math.degrees(calib.rpyCalib[1])
          yaw = math.degrees(calib.rpyCalib[2])
          desc += tr(" Your device is pointed {:.1f}° {} and {:.1f}° {}.").format(abs(pitch), tr("down") if pitch > 0 else tr("up"),
                                                                                  abs(yaw), tr("left") if yaw > 0 else tr("right"))
      except Exception:
        cloudlog.exception("invalid CalibrationParams")

    lag_perc = 0
    lag_bytes = self._params.get("LiveDelay")
    if lag_bytes:
      try:
        lag_perc = messaging.log_from_bytes(lag_bytes, log.Event).liveDelay.calPerc
      except Exception:
        cloudlog.exception("invalid LiveDelay")
    if lag_perc < 100:
      desc += tr("<br><br>Steering lag calibration is {}% complete.").format(lag_perc)
    else:
      desc += tr("<br><br>Steering lag calibration is complete.")

    torque_bytes = self._params.get("LiveTorqueParameters")
    if torque_bytes:
      try:
        torque = messaging.log_from_bytes(torque_bytes, log.Event).liveTorqueParameters
        # don't add for non-torque cars
        if torque.useParams:
          torque_perc = torque.calPerc
          if torque_perc < 100:
            desc += tr(" Steering torque response calibration is {}% complete.").format(torque_perc)
          else:
            desc += tr(" Steering torque response calibration is complete.")
      except Exception:
        cloudlog.exception("invalid LiveTorqueParameters")

    desc += "<br><br>"
    desc += tr("openpilot is continuously calibrating, resetting is rarely required. " +
               "Resetting calibration will restart openpilot if the car is powered on.")

    self._reset_calib_btn.set_description(desc)

  def _reboot_prompt(self):
    if ui_state.engaged:
      gui_app.set_modal_overlay(alert_dialog(tr("Disengage to Reboot")))
      return

    dialog = ConfirmDialog(tr("Are you sure you want to reboot?"), tr("Reboot"))
    gui_app.set_modal_overlay(dialog, callback=self._perform_reboot)

  def _perform_reboot(self, result: int):
    if not ui_state.engaged and result == DialogResult.CONFIRM:
      self._params.put_bool_nonblocking("DoReboot", True)

  def _power_off_prompt(self):
    if ui_state.engaged:
      gui_app.set_modal_overlay(alert_dialog(tr("Disengage to Power Off")))
      return

    dialog = ConfirmDialog(tr("Are you sure you want to power off?"), tr("Power Off"))
    gui_app.set_modal_overlay(dialog, callback=self._perform_power_off)

  def _perform_power_off(self, result: int):
    if not ui_state.engaged and result == DialogResult.CONFIRM:
      self._params.put_bool_nonblocking("DoShutdown", True)

  def _pair_device(self):
    if not self._pair_device_dialog:
      self._pair_device_dialog = PairingDialog()
    gui_app.set_modal_overlay(self._pair_device_dialog, callback=lambda result: setattr(self, '_pair_device_dialog', None))

  def _on_regulatory(self):
    if not self._fcc_dialog:
      self._fcc_dialog = HtmlModal(os.path.join(BASEDIR, "selfdrive/assets/offroad/fcc.html"))
    gui_app.set_modal_overlay(self._fcc_dialog)

  def _on_review_training_guide(self):
    if not self._training_guide:
      def completed_callback():
        gui_app.set_modal_overlay(None)

      self._training_guide = TrainingGuide(completed_callback=completed_callback)
    gui_app.set_modal_overlay(self._training_guide)
