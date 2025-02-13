import pyray as rl
import os
import sys
import time
import threading
from enum import IntEnum
from openpilot.system.ui.raylib.gui.button import gui_button
from openpilot.system.ui.raylib.gui.label import gui_label
from openpilot.system.ui.raylib.gui.application import gui_app


NVME = "/dev/nvme0n1"
USERDATA = "/dev/disk/by-partlabel/userdata"

def do_erase():
  # Best effort to wipe NVME
  # os.system(f"sudo umount {NVME}")
  # os.system(f"yes | sudo mkfs.ext4 {NVME}")

  # # Removing data and formatting
  # rm = os.system("sudo rm -rf /data/*")
  # os.system(f"sudo umount {USERDATA}")
  # fmt = os.system(f"yes | sudo mkfs.ext4 {USERDATA}")

  # if rm == 0 or fmt == 0:
  #   os.system("sudo reboot")
  return "Reset failed. Reboot to try again."

class ResetMode(IntEnum):
  USER_RESET = 0
  RECOVER = 1
  FORMAT = 2

class Reset:
  def __init__(self, mode):
    self.mode = mode
    self.confirm_msg = "Are you sure you want to reset your device?"
    self.body_text = "System reset triggered. Press confirm to erase all content and settings. Press cancel to resume boot."
    self.resetting = False

  def start_reset(self):
    self.body_text = "Resetting device...\nThis may take up to a minute."
    self.resetting = True
    threading.Timer(0.1, do_erase).start()

  def confirm(self):
    if self.body_text != self.confirm_msg:
      self.body_text = self.confirm_msg
    else:
      self.start_reset()

def draw_reset_ui(reset):
    rl.begin_drawing()
    rl.clear_background(rl.BLACK)
    left_margin, top_margin, right_margin, bottom_margin = 45, 220, 45, 45
    content_rect = rl.Rectangle(left_margin, top_margin,
                                rl.get_screen_width() - (left_margin + right_margin),
                                rl.get_screen_height() - (top_margin + bottom_margin))
    # Title
    label_rect = content_rect
    label_rect.x += 140
    label_rect.width -= 280
    gui_label(label_rect, "System Reset", 90)

    # Body text
    label_rect.y += 90 + 60
    gui_label(label_rect, reset.body_text, 80)

    # Buttons
    button_height = 160
    button_top = rl.get_screen_height() - button_height - bottom_margin
    button_width = (rl.get_screen_width() - (left_margin + right_margin + 50)) / 2.0
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_SIZE, 55)
    if not reset.resetting:
      if gui_button(rl.Rectangle(left_margin, button_top, button_width, button_height), "Cancel"):
        rl.close_window()
      if reset.resetting:
        if gui_button(rl.Rectangle(left_margin + button_width + 50, button_top, button_width, button_height), "Reboot"):
          os.system("sudo reboot")
      else:
        if gui_button(rl.Rectangle(left_margin + button_width + 50, button_top, button_width, button_height), "Confirm", rl.Color(70, 91, 234, 255)):
          reset.confirm()

    rl.end_drawing()


def main():
  gui_app.init_window("Reset", 20)

  mode = ResetMode.USER_RESET
  if len(sys.argv) > 1:
    if sys.argv[1] == "--recover":
      mode = ResetMode.RECOVER
    elif sys.argv[1] == "--format":
      mode = ResetMode.FORMAT

  reset = Reset(mode)

  while not rl.window_should_close():
    draw_reset_ui(reset)

  gui_app.close()


if __name__ == "__main__":
  main()
