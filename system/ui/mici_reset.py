#!/usr/bin/env python3
import os
import sys
import threading
import time
from enum import IntEnum

import pyray as rl

from openpilot.system.hardware import PC
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.slider import SmallSlider
from openpilot.system.ui.widgets.button import SmallButton, FullRoundedButton
from openpilot.system.ui.widgets.label import gui_label, gui_text_box

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


class Reset(Widget):
  def __init__(self, mode):
    super().__init__()
    self._mode = mode
    self._previous_reset_state = None
    self._reset_state = ResetState.NONE

    self._cancel_button = SmallButton("cancel")
    self._cancel_button.set_click_callback(self._cancel_callback)

    self._reboot_button = FullRoundedButton("reboot")
    self._reboot_button.set_click_callback(self._do_reboot)

    self._confirm_slider = SmallSlider("reset", self._confirm)

    self._render_status = True

  def _cancel_callback(self):
    self._render_status = False

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
    if self._reset_state != self._previous_reset_state:
      self._previous_reset_state = self._reset_state
      self._timeout_st = time.monotonic()
    elif self._reset_state != ResetState.RESETTING and (time.monotonic() - self._timeout_st) > TIMEOUT:
      exit(0)

  def _render(self, rect: rl.Rectangle):
    label_rect = rl.Rectangle(rect.x + 8, rect.y + 8, rect.width, 50)
    gui_label(label_rect, "factory reset", 48, font_weight=FontWeight.BOLD,
              color=rl.Color(255, 255, 255, int(255 * 0.9)))

    text_rect = rl.Rectangle(rect.x + 8, rect.y + 56, rect.width - 8 * 2, rect.height - 80)
    gui_text_box(text_rect, self._get_body_text(), 36, font_weight=FontWeight.ROMAN, line_scale=0.9)

    if self._reset_state != ResetState.RESETTING:
      # fade out cancel button as slider is moved, set visible to prevent pressing invisible cancel
      self._cancel_button.set_opacity(1.0 - self._confirm_slider.slider_percentage)
      self._cancel_button.set_visible(self._confirm_slider.slider_percentage < 0.8)

      if self._mode == ResetMode.RECOVER:
        self._cancel_button.set_text("reboot")
        self._cancel_button.render(rl.Rectangle(
          rect.x + 8,
          rect.y + rect.height - self._cancel_button.rect.height,
          self._cancel_button.rect.width,
          self._cancel_button.rect.height))
      elif self._mode == ResetMode.USER_RESET and self._reset_state != ResetState.FAILED:
        self._cancel_button.render(rl.Rectangle(
          rect.x + 8,
          rect.y + rect.height - self._cancel_button.rect.height,
          self._cancel_button.rect.width,
          self._cancel_button.rect.height))

      if self._reset_state != ResetState.FAILED:
        self._confirm_slider.render(rl.Rectangle(
          rect.x + rect.width - self._confirm_slider.rect.width,
          rect.y + rect.height - self._confirm_slider.rect.height,
          self._confirm_slider.rect.width,
          self._confirm_slider.rect.height))
      else:
        self._reboot_button.render(rl.Rectangle(
          rect.x + 8,
          rect.y + rect.height - self._reboot_button.rect.height,
          self._reboot_button.rect.width,
          self._reboot_button.rect.height))

    return self._render_status

  def _confirm(self):
    self.start_reset()

  def _get_body_text(self):
    if self._reset_state == ResetState.RESETTING:
      return "Resetting device... This may take up to a minute."
    if self._reset_state == ResetState.FAILED:
      return "Reset failed. Reboot to try again."
    if self._mode == ResetMode.RECOVER:
      return "Unable to mount data partition. It may be corrupted."
    return "All content and settings will be erased."


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

  for should_render in gui_app.render():
    if should_render:
      if not reset.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height)):
        break


if __name__ == "__main__":
  main()
