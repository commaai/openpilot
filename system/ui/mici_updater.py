#!/usr/bin/env python3
import sys
import subprocess
import threading
import pyray as rl
from enum import IntEnum

from openpilot.system.hardware import HARDWARE
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.wifi_manager import WifiManager, Network
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import gui_text_box, gui_label, UnifiedLabel
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
    self._current_network_strength = -1

    self.progress_value = 0
    self.progress_text = "loading"
    self.process = None
    self.update_thread = None
    self._wifi_manager = WifiManager()
    self._wifi_manager.set_active(True)

    self._network_setup_page = NetworkSetupPage(self._wifi_manager, self._network_setup_continue_callback,
                                                self._network_setup_back_callback)

    self._wifi_manager.add_callbacks(networks_updated=self._on_network_updated)
    self._network_monitor = NetworkConnectivityMonitor()
    self._network_monitor.start()

    # Buttons
    self._continue_button = FullRoundedButton("continue")
    self._continue_button.set_click_callback(lambda: self.set_current_screen(Screen.WIFI))

    self._title_label = UnifiedLabel("update required", 48, text_color=rl.Color(255, 115, 0, 255),
                                     font_weight=FontWeight.DISPLAY)
    self._subtitle_label = UnifiedLabel("The download size is approximately 1GB.", 36,
                                        text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                                        font_weight=FontWeight.ROMAN)

    self._update_failed_page = FailedPage(HARDWARE.reboot, self._update_failed_retry_callback,
                                          title="update failed")

  def _network_setup_back_callback(self):
    self.set_current_screen(Screen.PROMPT)

  def _network_setup_continue_callback(self):
    self.install_update()

  def _update_failed_retry_callback(self):
    self.set_current_screen(Screen.PROMPT)

  def _on_network_updated(self, networks: list[Network]):
    self._current_network_strength = next((net.strength for net in networks if net.is_connected), -1)

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
    title_rect = rl.Rectangle(self._rect.x + 6, self._rect.y - 5, self._rect.width - 12, self._rect.height - 8)
    if ' ' in self.progress_text:
      font_size = 62
    else:
      font_size = 82
    gui_text_box(title_rect, self.progress_text, font_size, font_weight=FontWeight.DISPLAY,
                 color=rl.Color(255, 255, 255, int(255 * 0.9)))

    progress_value = f"{self.progress_value}%"
    text_height = measure_text_cached(gui_app.font(FontWeight.ROMAN), progress_value, 128).y
    progress_rect = rl.Rectangle(self._rect.x + 6, self._rect.y + self._rect.height - text_height + 18,
                                 self._rect.width - 12, text_height)
    gui_label(progress_rect, progress_value, 128, font_weight=FontWeight.ROMAN,
              color=rl.Color(255, 255, 255, int(255 * 0.9 * 0.35)))

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
    for should_render in gui_app.render():
      if should_render:
        updater.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
    updater.close()
  except Exception as e:
    print(f"Updater error: {e}")
  finally:
    gui_app.close()


if __name__ == "__main__":
  main()
