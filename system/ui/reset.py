#!/usr/bin/env python3
import os
import pyray as rl
import sys
import threading
from enum import IntEnum

from openpilot.system.hardware import PC
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import gui_button, ButtonStyle
from openpilot.system.ui.widgets.label import Label, Text

NVME = "/dev/nvme0n1"
USERDATA = "/dev/disk/by-partlabel/userdata"

PADDING_HORIZONTAL = 140


class ResetMode(IntEnum):
  USER_RESET = 0  # user initiated a factory reset from openpilot
  RECOVER = 1  # userdata is corrupt for some reason, give a chance to recover
  FORMAT = 2  # finish up a factory reset from a tool that doesn't flash an empty partition to userdata


class ResetState(IntEnum):
  NONE = 0
  CONFIRM = 1
  RESETTING = 2
  FAILED = 3


class Reset(Widget):
  def __init__(self, mode):
    super().__init__()
    self.mode = mode
    self.reset_state = ResetState.NONE
    self.header_label = Label("System Reset", font_size=100, font_weight=FontWeight.BOLD)
    self.body_text = Text(self.get_body_text(), font_size=90)

  def _do_erase(self):
    if PC:
      return

    # Best effort to wipe NVME
    os.system(f"sudo umount {NVME}")
    os.system(f"yes | sudo mkfs.ext4 {NVME}")

    # Removing data and formatting
    rm = os.system("sudo rm -rf /data/*")
    os.system(f"sudo umount {USERDATA}")
    fmt = os.system(f"yes | sudo mkfs.ext4 {USERDATA}")

    if rm == 0 or fmt == 0:
      os.system("sudo reboot")
    else:
      self.reset_state = ResetState.FAILED

  def start_reset(self):
    self.reset_state = ResetState.RESETTING
    threading.Timer(0.1, self._do_erase).start()

  def _render(self, rect: rl.Rectangle):
    # Render header and text
    self.header_label.render(rl.Rectangle(rect.x + PADDING_HORIZONTAL, rect.y, rect.width - PADDING_HORIZONTAL * 2, self.header_label.font_size))
    self.body_text.text = self.get_body_text()
    self.body_text.render(
      rl.Rectangle(
        rect.x + PADDING_HORIZONTAL,
        rect.y + self.header_label.font_size + 40,
        rect.width - PADDING_HORIZONTAL * 2,
        rect.height,
      )
    )

    button_height = 160
    button_spacing = 50
    button_top = rect.y + rect.height - button_height
    button_width = (rect.width - button_spacing) / 2.0

    if self.reset_state != ResetState.RESETTING:
      if self.mode == ResetMode.RECOVER or self.reset_state == ResetState.FAILED:
        if gui_button(rl.Rectangle(rect.x, button_top, button_width, button_height), "Reboot"):
          os.system("sudo reboot")
      elif self.mode == ResetMode.USER_RESET:
        if gui_button(rl.Rectangle(rect.x, button_top, button_width, button_height), "Cancel"):
          return False

      if self.reset_state != ResetState.FAILED:
        if gui_button(rl.Rectangle(rect.x + button_width + 50, button_top, button_width, button_height), "Confirm", button_style=ButtonStyle.PRIMARY):
          self.confirm()

    return True

  def confirm(self):
    if self.reset_state == ResetState.CONFIRM:
      self.start_reset()
    else:
      self.reset_state = ResetState.CONFIRM

  def get_body_text(self):
    if self.reset_state == ResetState.CONFIRM:
      return "Are you sure you want to reset your device?"
    if self.reset_state == ResetState.RESETTING:
      return "Resetting device...\nThis may take up to a minute."
    if self.reset_state == ResetState.FAILED:
      return "Reset failed. Reboot to try again."
    if self.mode == ResetMode.RECOVER:
      return "Unable to mount data partition. Partition may be corrupted. Press confirm to erase and reset your device."
    return "System reset triggered. Press confirm to erase all content and settings. Press cancel to resume boot."


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

  for _ in gui_app.render():
    if not reset.render(rl.Rectangle(45, 200, gui_app.width - 90, gui_app.height - 245)):
      break


if __name__ == "__main__":
  main()
