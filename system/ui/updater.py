#!/usr/bin/env python3
import sys
import subprocess
import threading
import pyray as rl
from enum import IntEnum

from openpilot.system.hardware import HARDWARE
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.wifi_manager import WifiManagerWrapper
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import gui_button, ButtonStyle
from openpilot.system.ui.widgets.label import gui_text_box, gui_label
from openpilot.system.ui.widgets.network import WifiManagerUI

# Constants
MARGIN = 50
BUTTON_HEIGHT = 160
BUTTON_WIDTH = 400
PROGRESS_BAR_HEIGHT = 72
TITLE_FONT_SIZE = 80
BODY_FONT_SIZE = 65
BACKGROUND_COLOR = rl.BLACK
PROGRESS_BG_COLOR = rl.Color(41, 41, 41, 255)
PROGRESS_COLOR = rl.Color(54, 77, 239, 255)


class Screen(IntEnum):
  PROMPT = 0
  WIFI = 1
  PROGRESS = 2


class Updater(Widget):
  def __init__(self, updater_path, manifest_path):
    super().__init__()
    self.updater = updater_path
    self.manifest = manifest_path
    self.current_screen = Screen.PROMPT

    self.progress_value = 0
    self.progress_text = "Loading..."
    self.show_reboot_button = False
    self.process = None
    self.update_thread = None
    self.wifi_manager = WifiManagerWrapper()
    self.wifi_manager_ui = WifiManagerUI(self.wifi_manager)

  def install_update(self):
    self.current_screen = Screen.PROGRESS
    self.progress_value = 0
    self.progress_text = "Downloading..."
    self.show_reboot_button = False

    # Start the update process in a separate thread
    self.update_thread = threading.Thread(target=self._run_update_process)
    self.update_thread.daemon = True
    self.update_thread.start()

  def _run_update_process(self):
    # TODO: just import it and run in a thread without a subprocess
    cmd = [self.updater, "--swap", self.manifest]
    self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, bufsize=1, universal_newlines=True)

    for line in self.process.stdout:
      parts = line.strip().split(":")
      if len(parts) == 2:
        self.progress_text = parts[0]
        try:
          self.progress_value = int(float(parts[1]))
        except ValueError:
          pass

    exit_code = self.process.wait()
    if exit_code == 0:
      HARDWARE.reboot()
    else:
      self.progress_text = "Update failed"
      self.show_reboot_button = True

  def render_prompt_screen(self, rect: rl.Rectangle):
    # Title
    title_rect = rl.Rectangle(MARGIN + 50, 250, rect.width - MARGIN * 2 - 100, TITLE_FONT_SIZE)
    gui_label(title_rect, "Update Required", TITLE_FONT_SIZE, font_weight=FontWeight.BOLD)

    # Description
    desc_text = ("An operating system update is required. Connect your device to Wi-Fi for the fastest update experience. " +
                 "The download size is approximately 1GB.")

    desc_rect = rl.Rectangle(MARGIN + 50, 250 + TITLE_FONT_SIZE + 75, rect.width - MARGIN * 2 - 100, BODY_FONT_SIZE * 3)
    gui_text_box(desc_rect, desc_text, BODY_FONT_SIZE)

    # Buttons at the bottom
    button_y = rect.height - MARGIN - BUTTON_HEIGHT
    button_width = (rect.width - MARGIN * 3) // 2

    # WiFi button
    wifi_button_rect = rl.Rectangle(MARGIN, button_y, button_width, BUTTON_HEIGHT)
    if gui_button(wifi_button_rect, "Connect to Wi-Fi"):
      self.current_screen = Screen.WIFI
      return  # Return to avoid processing other buttons after screen change

    # Install button
    install_button_rect = rl.Rectangle(MARGIN * 2 + button_width, button_y, button_width, BUTTON_HEIGHT)
    if gui_button(install_button_rect, "Install", button_style=ButtonStyle.PRIMARY):
      self.install_update()
      return  # Return to avoid further processing after action

  def render_wifi_screen(self, rect: rl.Rectangle):
    # Draw the Wi-Fi manager UI
    wifi_rect = rl.Rectangle(MARGIN + 50, MARGIN, rect.width - MARGIN * 2 - 100, rect.height - MARGIN * 2 - BUTTON_HEIGHT - 20)
    self.wifi_manager_ui.render(wifi_rect)

    back_button_rect = rl.Rectangle(MARGIN, rect.height - MARGIN - BUTTON_HEIGHT, BUTTON_WIDTH, BUTTON_HEIGHT)
    if gui_button(back_button_rect, "Back"):
      self.current_screen = Screen.PROMPT
      return  # Return to avoid processing other interactions after screen change

  def render_progress_screen(self, rect: rl.Rectangle):
    title_rect = rl.Rectangle(MARGIN + 100, 330, rect.width - MARGIN * 2 - 200, 100)
    gui_label(title_rect, self.progress_text, 90, font_weight=FontWeight.SEMI_BOLD)

    # Progress bar
    bar_rect = rl.Rectangle(MARGIN + 100, 330 + 100 + 100, rect.width - MARGIN * 2 - 200, PROGRESS_BAR_HEIGHT)
    rl.draw_rectangle_rounded(bar_rect, 0.5, 10, PROGRESS_BG_COLOR)

    # Calculate the width of the progress chunk
    progress_width = (bar_rect.width * self.progress_value) / 100
    if progress_width > 0:
      progress_rect = rl.Rectangle(bar_rect.x, bar_rect.y, progress_width, bar_rect.height)
      rl.draw_rectangle_rounded(progress_rect, 0.5, 10, PROGRESS_COLOR)

    # Show reboot button if needed
    if self.show_reboot_button:
      reboot_rect = rl.Rectangle(MARGIN + 100, rect.height - MARGIN - BUTTON_HEIGHT, BUTTON_WIDTH, BUTTON_HEIGHT)
      if gui_button(reboot_rect, "Reboot"):
        # Return True to signal main loop to exit before rebooting
        HARDWARE.reboot()
        return

  def _render(self, rect: rl.Rectangle):
    if self.current_screen == Screen.PROMPT:
      self.render_prompt_screen(rect)
    elif self.current_screen == Screen.WIFI:
      self.render_wifi_screen(rect)
    elif self.current_screen == Screen.PROGRESS:
      self.render_progress_screen(rect)


def main():
  if len(sys.argv) < 3:
    print("Usage: updater.py <updater_path> <manifest_path>")
    sys.exit(1)

  updater_path = sys.argv[1]
  manifest_path = sys.argv[2]

  try:
    gui_app.init_window("System Update")
    updater = Updater(updater_path, manifest_path)
    for _ in gui_app.render():
      updater.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
  finally:
    # Make sure we clean up even if there's an error
    gui_app.close()


if __name__ == "__main__":
  main()
