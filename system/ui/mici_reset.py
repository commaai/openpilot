#!/usr/bin/env python3
import os
import sys
import threading
import time
from enum import IntEnum

import pyray as rl

from openpilot.system.hardware import PC
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.system.ui.mici_setup import GreyBigButton
from openpilot.selfdrive.ui.mici.widgets.button import BigCircleButton
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationDialogV2

USERDATA = "/dev/disk/by-partlabel/userdata"
TIMEOUT = 3*60


class ResetMode(IntEnum):
  USER_RESET = 0  # user initiated a factory reset from openpilot
  RECOVER = 1     # userdata is corrupt for some reason, give a chance to recover
  FORMAT = 2      # finish up a factory reset from a tool that doesn't flash an empty partition to userdata


class ResetState(IntEnum):
  NONE = 0
  RESETTING = 1
  FAILED = 2


class Reset(Scroller):
  def __init__(self, mode):
    super().__init__()
    self._mode = mode
    self._previous_reset_state = None
    self._reset_state = ResetState.NONE

    self._info_button = GreyBigButton("factory reset", self._get_body_text(),
                                      gui_app.texture("icons_mici/setup/factory_reset.png", 64, 64))

    # red circle button: confirm reset (launches slide-to-reset dialog)
    self._reset_button = BigCircleButton("icons_mici/settings/device/uninstall.png", red=True, icon_size=(64, 64))
    self._reset_button.set_click_callback(self._on_reset_click)

    # black circle button: cancel (USER_RESET) or reboot (RECOVER)
    self._cancel_button = BigCircleButton("icons_mici/settings/device/power.png", red=False, icon_size=(64, 66))
    self._cancel_button.set_click_callback(self._on_cancel_click)

    # reboot button shown only in FAILED state
    self._reboot_button = BigCircleButton("icons_mici/settings/device/reboot.png", red=False, icon_size=(64, 70))
    self._reboot_button.set_click_callback(self._on_reboot_click)
    self._reboot_button.set_visible(False)

    self._scroller.add_widgets([
      self._info_button,
      self._reset_button,
      self._cancel_button,
      self._reboot_button,
    ])

  def _on_reset_click(self):
    dlg = BigConfirmationDialogV2("erase\ndevice", "icons_mici/settings/device/uninstall.png", red=True,
                                  confirm_callback=self._confirm)
    gui_app.push_widget(dlg)

  def _on_cancel_click(self):
    if self._mode == ResetMode.RECOVER:
      self._on_reboot_click()
    else:
      dlg = BigConfirmationDialogV2("normal\nstartup", "icons_mici/settings/device/power.png", red=False,
                                    confirm_callback=gui_app.request_close)
      gui_app.push_widget(dlg)

  def _on_reboot_click(self):
    dlg = BigConfirmationDialogV2("reboot\ndevice", "icons_mici/settings/device/reboot.png", red=False,
                                  confirm_callback=self._do_reboot)
    gui_app.push_widget(dlg)

  def _do_reboot(self):
    if PC:
      return
    os.system("sudo reboot")

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
      self._reset_state = ResetState.FAILED

  def start_reset(self):
    self._reset_state = ResetState.RESETTING
    threading.Timer(0.1, self._do_erase).start()

  def _update_state(self):
    super()._update_state()

    if self._reset_state != self._previous_reset_state:
      self._previous_reset_state = self._reset_state
      self._timeout_st = time.monotonic()
      self._info_button.set_value(self._get_body_text())

      if self._reset_state == ResetState.RESETTING:
        self._reset_button.set_visible(False)
        self._cancel_button.set_visible(False)
        self._reboot_button.set_visible(False)
      elif self._reset_state == ResetState.FAILED:
        self._reset_button.set_visible(False)
        # in RECOVER mode, keep cancel (reboot) visible; in USER_RESET, hide it
        self._cancel_button.set_visible(self._mode == ResetMode.RECOVER)
        self._reboot_button.set_visible(True)

    elif self._mode != ResetMode.RECOVER and self._reset_state != ResetState.RESETTING and (time.monotonic() - self._timeout_st) > TIMEOUT:
      exit(0)

  def _confirm(self):
    self.start_reset()

  def _get_body_text(self):
    if self._reset_state == ResetState.RESETTING:
      return "resetting device...\nthis may take up to a minute."
    if self._reset_state == ResetState.FAILED:
      return "reset failed.\nreboot to try again."
    if self._mode == ResetMode.RECOVER:
      return "unable to mount data partition.\nit may be corrupted."
    return "all content and\nsettings will be erased"


def main():
  mode = ResetMode.USER_RESET
  if len(sys.argv) > 1:
    if sys.argv[1] == '--recover':
      mode = ResetMode.RECOVER
    elif sys.argv[1] == "--format":
      mode = ResetMode.FORMAT

  gui_app.init_window("System Reset")
  reset = Reset(mode)

  if mode == ResetMode.FORMAT:
    reset.start_reset()

  gui_app.push_widget(reset)

  for _ in gui_app.render():
    pass


if __name__ == "__main__":
  main()
