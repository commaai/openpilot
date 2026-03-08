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
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationCircleButton

USERDATA = "/dev/disk/by-partlabel/userdata"
TIMEOUT = 3*60


class ResetMode(IntEnum):
  USER_RESET = 0  # user initiated a factory reset from openpilot
  RECOVER = 1     # userdata is corrupt for some reason, give a chance to recover
  TAP_RESET = 2   # user initiated a factory reset by tapping the screen during boot


class ResetFailedPage(FailedPage):
  def __init__(self):
    super().__init__(None, "reset failed", "reboot to try again", icon="icons_mici/setup/reset_failed.png")

  def show_event(self):
    super().show_event()
    self._nav_bar._alpha = 0.0  # not dismissable

  def _back_enabled(self) -> bool:
    return False


class ResettingPage(NavWidget):
  DOT_STEP = 0.6

  def __init__(self):
    super().__init__()
    self._show_time = 0.0

    self._resetting_card = GreyBigButton("resetting device", "this may take up to\na minute...",
                                         gui_app.texture("icons_mici/setup/factory_reset.png", 64, 64))

  def show_event(self):
    super().show_event()
    self._nav_bar._alpha = 0.0  # not dismissable
    self._show_time = rl.get_time()

  def _back_enabled(self) -> bool:
    return False

  def _render(self, _):
    t = (rl.get_time() - self._show_time) % (self.DOT_STEP * 2)
    dots = "." * min(int(t / (self.DOT_STEP / 4)), 3)
    self._resetting_card.set_value(f"this may take up to\na minute{dots}")
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

    self._reset_button = BigConfirmationCircleButton("reset &\nerase", gui_app.texture("icons_mici/settings/device/uninstall.png", 70, 70),
                                                     self._start_reset, exit_on_confirm=False, red=True)
    self._cancel_button = BigConfirmationCircleButton("cancel", gui_app.texture("icons_mici/setup/cancel.png", 64, 64),
                                                      gui_app.request_close, exit_on_confirm=False)
    self._reboot_button = BigConfirmationCircleButton("reboot\ndevice", gui_app.texture("icons_mici/settings/device/reboot.png", 64, 70),
                                                      HARDWARE.reboot, exit_on_confirm=False)

    # show reboot button if in recover mode
    self._cancel_button.set_visible(mode != ResetMode.RECOVER)
    self._reboot_button.set_visible(mode == ResetMode.RECOVER)

    main_card = GreyBigButton("factory reset", "resetting erases\nall user content & data",
                              gui_app.texture("icons_mici/setup/factory_reset.png", 64, 64))
    self._scroller.add_widget(main_card)

    if mode != ResetMode.USER_RESET:
      self._scroller.add_widget(GreyBigButton("", "Resetting erases all user content & data."))
      if mode == ResetMode.RECOVER:
        main_card.set_value("user data partition\ncould not be mounted")
      elif mode == ResetMode.TAP_RESET:
        main_card.set_value("reset triggered by\ntapping the screen")

    self._scroller.add_widgets([
      GreyBigButton("", "For a deeper reset, go to\nhttps://flash.comma.ai"),
      self._cancel_button,
      self._reboot_button,
      self._reset_button,
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

  def _start_reset(self):
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
    elif sys.argv[1] == '--tap-reset':
      mode = ResetMode.TAP_RESET

  gui_app.init_window("System Reset")
  reset = Reset(mode)
  gui_app.push_widget(reset)

  for _ in gui_app.render():
    pass


if __name__ == "__main__":
  main()
