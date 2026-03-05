#!/usr/bin/env python3
import os
import sys
import time
from enum import IntEnum

import pyray as rl

from openpilot.system.hardware import HARDWARE, PC
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.system.ui.widgets.nav_widget import NavWidget
from openpilot.system.ui.mici_setup import GreyBigButton, FailedPage
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationDialogV2
from openpilot.selfdrive.ui.mici.widgets.button import BigCircleButton

USERDATA = "/dev/disk/by-partlabel/userdata"
TIMEOUT = 3*60


class ResetMode(IntEnum):
  USER_RESET = 0  # user initiated a factory reset from openpilot
  RECOVER = 1     # userdata is corrupt for some reason, give a chance to recover
  FORMAT = 2      # finish up a factory reset from a tool that doesn't flash an empty partition to userdata


class ResetFailedPage(FailedPage):
  def __init__(self):
    super().__init__(None, "reset failed", "reboot to try again", icon="icons_mici/setup/reset_failed.png")

  def show_event(self):
    super().show_event()
    self._nav_bar._alpha = 0.0  # not dismissable

  def _back_enabled(self) -> bool:
    return False


class ResettingPage(NavWidget):
  def __init__(self):
    super().__init__()

    self._resetting_card = GreyBigButton("resetting device", "this may take up to\na minute...",
                                         gui_app.texture("icons_mici/setup/factory_reset.png", 64, 64))

  def show_event(self):
    super().show_event()
    self._nav_bar._alpha = 0.0  # not dismissable

  def _back_enabled(self) -> bool:
    return False

  def _render(self, _):
    self._resetting_card.render(rl.Rectangle(
      self._rect.x + self._rect.width / 2 - self._resetting_card.rect.width / 2,
      self._rect.y + self._rect.height / 2 - self._resetting_card.rect.height / 2,
      self._resetting_card.rect.width,
      self._resetting_card.rect.height,
    ))


class Reset(Scroller):
  def __init__(self, mode):
    super().__init__()
    self._mode = mode
    self._previous_active_widget = None
    self._reset_failed = False
    self._timeout_st = time.monotonic()

    self._resetting_page = ResettingPage()
    self._reset_failed_page = ResetFailedPage()

    def show_confirm_dialog():
      dialog = BigConfirmationDialogV2("erase\ndevice", "icons_mici/settings/device/uninstall.png", red=True,
                                       confirm_callback=self.start_reset)
      gui_app.push_widget(dialog)

    def show_cancel_dialog():
      dialog = BigConfirmationDialogV2("normal\nstartup", "icons_mici/settings/device/reboot.png",
                                       exit_on_confirm=False, confirm_callback=gui_app.request_close)
      gui_app.push_widget(dialog)

    def show_reboot_dialog():
      dialog = BigConfirmationDialogV2("reboot\ndevice", "icons_mici/settings/device/reboot.png",
                                       exit_on_confirm=False, confirm_callback=HARDWARE.reboot)
      gui_app.push_widget(dialog)

    self._reset_button = BigCircleButton("icons_mici/settings/device/uninstall.png", red=True)
    self._reset_button.set_click_callback(show_confirm_dialog)

    self._cancel_button = BigCircleButton("icons_mici/settings/device/reboot.png")
    self._cancel_button.set_click_callback(show_cancel_dialog)

    main_card = GreyBigButton("factory reset", "all content and\nsettings will be erased",
                              gui_app.texture("icons_mici/setup/factory_reset.png", 64, 64))

    # cancel button becomes reboot button
    if mode == ResetMode.RECOVER:
      main_card.set_text("unable to mount\ndata partition")
      main_card.set_value("it may be corrupted")
      self._cancel_button.set_click_callback(show_reboot_dialog)

    self._scroller.add_widgets([
      main_card,
      self._reset_button,
      self._cancel_button,
    ])

  def _do_erase(self):
    if PC:
      return

    # Removing data and formatting
    rm = os.system("sudo rm -rf /data/*")
    os.system(f"sudo umount {USERDATA}")
    fmt = os.system(f"yes | sudo mkfs.ext4 {USERDATA}")

    if rm == 0 or fmt == 0:
      os.system("sudo reboot")
    else:
      self._reset_failed = True

  def start_reset(self):
    self._resetting_page.set_shown_callback(self._do_erase)
    gui_app.push_widget(self._resetting_page)

  def _update_state(self):
    super()._update_state()

    if self._reset_failed:
      self._reset_failed = False
      gui_app.pop_widgets_to(self, lambda: gui_app.push_widget(self._reset_failed_page))

    active_widget = gui_app.get_active_widget()
    if active_widget != self._previous_active_widget:
      self._previous_active_widget = active_widget
      self._timeout_st = time.monotonic()
    elif self._mode != ResetMode.RECOVER and active_widget != self._resetting_page and (time.monotonic() - self._timeout_st) > TIMEOUT:
      exit(0)


def main():
  mode = ResetMode.USER_RESET
  if len(sys.argv) > 1:
    if sys.argv[1] == '--recover':
      mode = ResetMode.RECOVER
    elif sys.argv[1] == "--format":
      mode = ResetMode.FORMAT

  gui_app.init_window("System Reset")
  reset = Reset(mode)
  gui_app.push_widget(reset)

  if mode == ResetMode.FORMAT:
    reset.start_reset()

  for _ in gui_app.render():
    pass


if __name__ == "__main__":
  main()
