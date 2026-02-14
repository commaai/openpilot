import os
import threading
import pyray as rl
from enum import IntEnum
from collections.abc import Callable

from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.time_helpers import system_time_valid
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.system.ui.lib.scroll_panel2 import GuiScrollPanel2
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigCircleButton
from openpilot.selfdrive.ui.mici.widgets.dialog import BigDialog, BigConfirmationDialogV2
from openpilot.selfdrive.ui.mici.widgets.pairing_dialog import PairingDialog
from openpilot.selfdrive.ui.mici.onroad.driver_camera_dialog import DriverCameraDialog
from openpilot.selfdrive.ui.mici.layouts.onboarding import TrainingGuide
from openpilot.system.ui.lib.application import gui_app, FontWeight, MousePos
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import Widget, NavWidget
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.widgets.label import MiciLabel
from openpilot.system.ui.widgets.html_render import HtmlModal, HtmlRenderer
from openpilot.system.athena.registration import UNREGISTERED_DONGLE_ID


class MiciFccModal(NavWidget):
  BACK_TOUCH_AREA_PERCENTAGE = 0.1

  def __init__(self, file_path: str | None = None, text: str | None = None):
    super().__init__()
    self.set_back_callback(lambda: gui_app.set_modal_overlay(None))
    self._content = HtmlRenderer(file_path=file_path, text=text)
    self._scroll_panel = GuiScrollPanel2(horizontal=False)
    self._fcc_logo = gui_app.texture("icons_mici/settings/device/fcc_logo.png", 76, 64)

  def _render(self, rect: rl.Rectangle):
    content_height = self._content.get_total_height(int(rect.width))
    content_height += self._fcc_logo.height + 20

    scroll_content_rect = rl.Rectangle(rect.x, rect.y, rect.width, content_height)
    scroll_offset = round(self._scroll_panel.update(rect, scroll_content_rect.height))

    fcc_pos = rl.Vector2(rect.x + 20, rect.y + 20 + scroll_offset)

    scroll_content_rect.y += scroll_offset + self._fcc_logo.height + 20
    self._content.render(scroll_content_rect)

    rl.draw_texture_ex(self._fcc_logo, fcc_pos, 0.0, 1.0, rl.WHITE)

    return -1


def _engaged_confirmation_callback(callback: Callable, action_text: str):
  if not ui_state.engaged:
    def confirm_callback():
      # Check engaged again in case it changed while the dialog was open
      if not ui_state.engaged:
        callback()

    red = False
    if action_text == "power off":
      icon = "icons_mici/settings/device/power.png"
      red = True
    elif action_text == "reboot":
      icon = "icons_mici/settings/device/reboot.png"
    elif action_text == "reset":
      icon = "icons_mici/settings/device/lkas.png"
    elif action_text == "uninstall":
      icon = "icons_mici/settings/device/uninstall.png"
    else:
      # TODO: check
      icon = "icons_mici/settings/comma_icon.png"

    dlg: BigConfirmationDialogV2 | BigDialog = BigConfirmationDialogV2(f"slide to\n{action_text.lower()}", icon, red=red,
                                                                       exit_on_confirm=action_text == "reset",
                                                                       confirm_callback=confirm_callback)
    gui_app.set_modal_overlay(dlg)
  else:
    dlg = BigDialog(f"Disengage to {action_text}", "")
    gui_app.set_modal_overlay(dlg)


class DeviceInfoLayoutMici(Widget):
  def __init__(self):
    super().__init__()

    self.set_rect(rl.Rectangle(0, 0, 360, 180))

    params = Params()
    header_color = rl.Color(255, 255, 255, int(255 * 0.9))
    subheader_color = rl.Color(255, 255, 255, int(255 * 0.9 * 0.65))
    max_width = int(self._rect.width - 20)
    self._dongle_id_label = MiciLabel("device ID", 48, width=max_width, color=header_color, font_weight=FontWeight.DISPLAY)
    self._dongle_id_text_label = MiciLabel(params.get("DongleId") or 'N/A', 32, width=max_width, color=subheader_color, font_weight=FontWeight.ROMAN)

    self._serial_number_label = MiciLabel("serial", 48, color=header_color, font_weight=FontWeight.DISPLAY)
    self._serial_number_text_label = MiciLabel(params.get("HardwareSerial") or 'N/A', 32, width=max_width, color=subheader_color, font_weight=FontWeight.ROMAN)

  def _render(self, _):
    self._dongle_id_label.set_position(self._rect.x + 20, self._rect.y - 10)
    self._dongle_id_label.render()

    self._dongle_id_text_label.set_position(self._rect.x + 20, self._rect.y + 68 - 25)
    self._dongle_id_text_label.render()

    self._serial_number_label.set_position(self._rect.x + 20, self._rect.y + 114 - 30)
    self._serial_number_label.render()

    self._serial_number_text_label.set_position(self._rect.x + 20, self._rect.y + 161 - 25)
    self._serial_number_text_label.render()


class UpdaterState(IntEnum):
  IDLE = 0
  WAITING_FOR_UPDATER = 1
  UPDATER_RESPONDING = 2


class PairBigButton(BigButton):
  def __init__(self):
    super().__init__("pair", "connect.comma.ai", "icons_mici/settings/comma_icon.png", icon_size=(33, 60))

  def _get_label_font_size(self):
    return 64

  def _update_state(self):
    if ui_state.prime_state.is_paired():
      self.set_text("paired")
      if ui_state.prime_state.is_prime():
        self.set_value("subscribed")
      else:
        self.set_value("upgrade to prime")
    else:
      self.set_text("pair")
      self.set_value("connect.comma.ai")

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)

    # TODO: show ad dialog when clicked if not prime
    if ui_state.prime_state.is_paired():
      return
    dlg: BigDialog | PairingDialog
    if not system_time_valid():
      dlg = BigDialog(tr("Please connect to Wi-Fi to complete initial pairing"), "")
    elif UNREGISTERED_DONGLE_ID == (ui_state.params.get("DongleId") or UNREGISTERED_DONGLE_ID):
      dlg = BigDialog(tr("Device must be registered with the comma.ai backend to pair"), "")
    else:
      dlg = PairingDialog()
    gui_app.set_modal_overlay(dlg)


UPDATER_TIMEOUT = 10.0  # seconds to wait for updater to respond


class UpdateOpenpilotBigButton(BigButton):
  def __init__(self):
    self._txt_update_icon = gui_app.texture("icons_mici/settings/device/update.png", 64, 75)
    self._txt_reboot_icon = gui_app.texture("icons_mici/settings/device/reboot.png", 64, 70)
    self._txt_up_to_date_icon = gui_app.texture("icons_mici/settings/device/up_to_date.png", 64, 64)
    super().__init__("update openpilot", "", self._txt_update_icon)

    self._waiting_for_updater_t: float | None = None
    self._hide_value_t: float | None = None
    self._state: UpdaterState = UpdaterState.IDLE

    ui_state.add_offroad_transition_callback(self.offroad_transition)

  def offroad_transition(self):
    if ui_state.is_offroad():
      self.set_enabled(True)

  def _handle_mouse_release(self, mouse_pos: MousePos):
    if not system_time_valid():
      dlg = BigDialog(tr("Please connect to Wi-Fi to update"), "")
      gui_app.set_modal_overlay(dlg)
      return

    self.set_enabled(False)
    self._state = UpdaterState.WAITING_FOR_UPDATER
    self.set_icon(self._txt_update_icon)

    def run():
      if self.get_value() == "download update":
        os.system("pkill -SIGHUP -f system.updated.updated")
      elif self.get_value() == "update now":
        ui_state.params.put_bool("DoReboot", True)
      else:
        os.system("pkill -SIGUSR1 -f system.updated.updated")

    threading.Thread(target=run, daemon=True).start()

  def set_value(self, value: str):
    super().set_value(value)
    if value:
      self.set_text("")
    else:
      self.set_text("update openpilot")

  def _update_state(self):
    if ui_state.started:
      self.set_enabled(False)
      return

    updater_state = ui_state.params.get("UpdaterState") or ""
    failed_count = ui_state.params.get("UpdateFailedCount")
    failed = False if failed_count is None else int(failed_count) > 0

    if ui_state.params.get_bool("UpdateAvailable"):
      self.set_rotate_icon(False)
      self.set_enabled(True)
      if self.get_value() != "update now":
        self.set_value("update now")
        self.set_icon(self._txt_reboot_icon)

    elif self._state == UpdaterState.WAITING_FOR_UPDATER:
      self.set_rotate_icon(True)
      if updater_state != "idle":
        self._state = UpdaterState.UPDATER_RESPONDING

      # Recover from updater not responding (time invalid shortly after boot)
      if self._waiting_for_updater_t is None:
        self._waiting_for_updater_t = rl.get_time()

      if self._waiting_for_updater_t is not None and rl.get_time() - self._waiting_for_updater_t > UPDATER_TIMEOUT:
        self.set_rotate_icon(False)
        self.set_value("updater failed\nto respond")
        self._state = UpdaterState.IDLE
        self._hide_value_t = rl.get_time()

    elif self._state == UpdaterState.UPDATER_RESPONDING:
      if updater_state == "idle":
        self.set_rotate_icon(False)
        self._state = UpdaterState.IDLE
        self._hide_value_t = rl.get_time()
      else:
        if self.get_value() != updater_state:
          self.set_value(updater_state)

    elif self._state == UpdaterState.IDLE:
      self.set_rotate_icon(False)
      if failed:
        if self.get_value() != "failed to update":
          self.set_value("failed to update")

      elif ui_state.params.get_bool("UpdaterFetchAvailable"):
        self.set_enabled(True)
        if self.get_value() != "download update":
          self.set_value("download update")

      elif self._hide_value_t is not None:
        self.set_enabled(True)
        if self.get_value() == "checking...":
          self.set_value("up to date")
          self.set_icon(self._txt_up_to_date_icon)

        # Hide previous text after short amount of time (up to date or failed)
        if rl.get_time() - self._hide_value_t > 3.0:
          self._hide_value_t = None
          self.set_value("")
          self.set_icon(self._txt_update_icon)
      else:
        if self.get_value() != "":
          self.set_value("")

    if self._state != UpdaterState.WAITING_FOR_UPDATER:
      self._waiting_for_updater_t = None


class DeviceLayoutMici(NavWidget):
  def __init__(self, back_callback: Callable):
    super().__init__()

    self._fcc_dialog: HtmlModal | None = None
    self._driver_camera: DriverCameraDialog | None = None
    self._training_guide: TrainingGuide | None = None

    def power_off_callback():
      ui_state.params.put_bool("DoShutdown", True)

    def reboot_callback():
      ui_state.params.put_bool("DoReboot", True)

    def reset_calibration_callback():
      params = ui_state.params
      params.remove("CalibrationParams")
      params.remove("LiveTorqueParameters")
      params.remove("LiveParameters")
      params.remove("LiveParametersV2")
      params.remove("LiveDelay")
      params.put_bool("OnroadCycleRequested", True)

    def uninstall_openpilot_callback():
      ui_state.params.put_bool("DoUninstall", True)

    reset_calibration_btn = BigButton("reset calibration", "", "icons_mici/settings/device/lkas.png", icon_size=(114, 60))
    reset_calibration_btn.set_click_callback(lambda: _engaged_confirmation_callback(reset_calibration_callback, "reset"))

    uninstall_openpilot_btn = BigButton("uninstall openpilot", "", "icons_mici/settings/device/uninstall.png")
    uninstall_openpilot_btn.set_click_callback(lambda: _engaged_confirmation_callback(uninstall_openpilot_callback, "uninstall"))

    reboot_btn = BigCircleButton("icons_mici/settings/device/reboot.png", red=False, icon_size=(64, 70))
    reboot_btn.set_click_callback(lambda: _engaged_confirmation_callback(reboot_callback, "reboot"))

    self._power_off_btn = BigCircleButton("icons_mici/settings/device/power.png", red=True, icon_size=(64, 66))
    self._power_off_btn.set_click_callback(lambda: _engaged_confirmation_callback(power_off_callback, "power off"))

    regulatory_btn = BigButton("regulatory info", "", "icons_mici/settings/device/info.png")
    regulatory_btn.set_click_callback(self._on_regulatory)

    driver_cam_btn = BigButton("driver\ncamera preview", "", "icons_mici/settings/device/cameras.png")
    driver_cam_btn.set_click_callback(self._show_driver_camera)
    driver_cam_btn.set_enabled(lambda: ui_state.is_offroad())

    review_training_guide_btn = BigButton("review\ntraining guide", "", "icons_mici/settings/device/info.png")
    review_training_guide_btn.set_click_callback(self._on_review_training_guide)
    review_training_guide_btn.set_enabled(lambda: ui_state.is_offroad())

    self._scroller = Scroller([
      DeviceInfoLayoutMici(),
      UpdateOpenpilotBigButton(),
      PairBigButton(),
      review_training_guide_btn,
      driver_cam_btn,
      reset_calibration_btn,
      uninstall_openpilot_btn,
      regulatory_btn,
      reboot_btn,
      self._power_off_btn,
    ], snap_items=False)

    # Set up back navigation
    self.set_back_callback(back_callback)

    # Hide power off button when onroad
    ui_state.add_offroad_transition_callback(self._offroad_transition)

  def _on_regulatory(self):
    if not self._fcc_dialog:
      self._fcc_dialog = MiciFccModal(os.path.join(BASEDIR, "selfdrive/assets/offroad/mici_fcc.html"))
    gui_app.set_modal_overlay(self._fcc_dialog)

  def _offroad_transition(self):
    self._power_off_btn.set_visible(ui_state.is_offroad())

  def _show_driver_camera(self):
    if not self._driver_camera:
      self._driver_camera = DriverCameraDialog()
    gui_app.set_modal_overlay(self._driver_camera, callback=lambda result: setattr(self, '_driver_camera', None))

  def _on_review_training_guide(self):
    if not self._training_guide:
      def completed_callback():
        gui_app.set_modal_overlay(None)

      self._training_guide = TrainingGuide(completed_callback=completed_callback)
    gui_app.set_modal_overlay(self._training_guide, callback=lambda result: setattr(self, '_training_guide', None))

  def show_event(self):
    super().show_event()
    self._scroller.show_event()

  def _render(self, rect: rl.Rectangle):
    self._scroller.render(rect)
