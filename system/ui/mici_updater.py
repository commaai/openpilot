#!/usr/bin/env python3
import sys
import subprocess
import threading
import pyray as rl
from enum import IntEnum

from openpilot.system.hardware import HARDWARE
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.wifi_manager import WifiManager
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets.button import FullRoundedButton
from openpilot.system.ui.mici_setup import NetworkSetupPage, FailedPage, NetworkConnectivityMonitor


class Screen(IntEnum):
  PROMPT = 0
  WIFI = 1
  PROGRESS = 2
  FAILED = 3


class Updater(Widget):
  def __init__(self, updater_path, manifest_path):
    super().__init__()
    self.updater = updater_path
    self.manifest = manifest_path
    self.current_screen = Screen.PROMPT

    self.progress_value = 0
    self.progress_text = "loading"
    self.process = None
    self.update_thread = None
    self._wifi_manager = WifiManager()
    self._wifi_manager.set_active(True)

    self._network_setup_page = NetworkSetupPage(self._wifi_manager, self._network_setup_continue_callback,
                                                self._network_setup_back_callback)
    self._network_setup_page.set_enabled(lambda: self.enabled)  # for nav stack

    self._network_monitor = NetworkConnectivityMonitor()
    self._network_monitor.start()

    # Buttons
    self._continue_button = FullRoundedButton("continue")
    self._continue_button.set_click_callback(lambda: self.set_current_screen(Screen.WIFI))

    self._title_label = UnifiedLabel("update required", 48, text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                                     font_weight=FontWeight.DISPLAY)
    self._subtitle_label = UnifiedLabel("The download size is approximately 1GB.", 36,
                                        text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                                        font_weight=FontWeight.ROMAN)

    self._update_failed_page = FailedPage(HARDWARE.reboot, self._update_failed_retry_callback,
                                          title="update failed")

    self._progress_title_label = UnifiedLabel("", 64, text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                                              font_weight=FontWeight.DISPLAY, line_height=0.8)
    self._progress_percent_label = UnifiedLabel("", 132, text_color=rl.Color(255, 255, 255, int(255 * 0.9 * 0.65)),
                                                font_weight=FontWeight.ROMAN,
                                                alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_BOTTOM)

  def _network_setup_back_callback(self):
    self.set_current_screen(Screen.PROMPT)

  def _network_setup_continue_callback(self):
    self.install_update()

  def _update_failed_retry_callback(self):
    self.set_current_screen(Screen.PROMPT)

  def set_current_screen(self, screen: Screen):
    if self.current_screen != screen:
      if screen == Screen.PROGRESS:
        if self._network_setup_page:
          self._network_setup_page.hide_event()
      elif screen == Screen.WIFI:
        if self._network_setup_page:
          self._network_setup_page.show_event()
      elif screen == Screen.PROMPT:
        if self._network_setup_page:
          self._network_setup_page.hide_event()
      elif screen == Screen.FAILED:
        if self._network_setup_page:
          self._network_setup_page.hide_event()

    self.current_screen = screen

  def install_update(self):
    self.set_current_screen(Screen.PROGRESS)
    self.progress_value = 0
    self.progress_text = "downloading"

    # Start the update process in a separate thread
    self.update_thread = threading.Thread(target=self._run_update_process)
    self.update_thread.daemon = True
    self.update_thread.start()

  def _run_update_process(self):
    # TODO: just import it and run in a thread without a subprocess
    cmd = [self.updater, "--swap", self.manifest]
    self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, bufsize=1, universal_newlines=True)

    if self.process.stdout is not None:
      for line in self.process.stdout:
        parts = line.strip().split(":")
        if len(parts) == 2:
          self.progress_text = parts[0].lower()
          try:
            self.progress_value = int(float(parts[1]))
          except ValueError:
            pass

    exit_code = self.process.wait()
    if exit_code == 0:
      HARDWARE.reboot()
    else:
      self.set_current_screen(Screen.FAILED)

  def render_prompt_screen(self, rect: rl.Rectangle):
    self._title_label.render(rl.Rectangle(
      rect.x + 8,
      rect.y - 5,
      rect.width,
      48,
    ))

    subtitle_width = rect.width - 16
    subtitle_height = self._subtitle_label.get_content_height(int(subtitle_width))
    self._subtitle_label.render(rl.Rectangle(
      rect.x + 8,
      rect.y + 48,
      subtitle_width,
      subtitle_height,
    ))

    self._continue_button.render(rl.Rectangle(
      rect.x + 8,
      rect.y + rect.height - self._continue_button.rect.height,
      self._continue_button.rect.width,
      self._continue_button.rect.height,
    ))

  def render_progress_screen(self, rect: rl.Rectangle):
    self._progress_title_label.set_text(self.progress_text.replace("_", "_\n") + "...")
    self._progress_title_label.render(rl.Rectangle(
      rect.x + 12,
      rect.y + 2,
      rect.width,
      self._progress_title_label.get_content_height(int(rect.width - 20)),
    ))

    self._progress_percent_label.set_text(f"{self.progress_value}%")
    self._progress_percent_label.render(rl.Rectangle(
      rect.x + 12,
      rect.y + 18,
      rect.width,
      rect.height,
    ))

  def _update_state(self):
    self._wifi_manager.process_callbacks()

  def _render(self, rect: rl.Rectangle):
    if self.current_screen == Screen.PROMPT:
      self.render_prompt_screen(rect)
    elif self.current_screen == Screen.WIFI:
      self._network_setup_page.set_has_internet(self._network_monitor.network_connected.is_set())
      self._network_setup_page.render(rect)
    elif self.current_screen == Screen.PROGRESS:
      self.render_progress_screen(rect)
    elif self.current_screen == Screen.FAILED:
      self._update_failed_page.render(rect)

  def close(self):
    self._network_monitor.stop()


def main():
  if len(sys.argv) < 3:
    print("Usage: updater.py <updater_path> <manifest_path>")
    sys.exit(1)

  updater_path = sys.argv[1]
  manifest_path = sys.argv[2]

  try:
    gui_app.init_window("System Update")
    updater = Updater(updater_path, manifest_path)
    gui_app.push_widget(updater)
    for _ in gui_app.render():
      pass
    updater.close()
  except Exception as e:
    print(f"Updater error: {e}")
  finally:
    gui_app.close()


if __name__ == "__main__":
  main()
