#!/usr/bin/env python3
import sys
import subprocess
import threading
import pyray as rl

from openpilot.common.realtime import config_realtime_process, set_core_affinity
from openpilot.system.hardware import HARDWARE, TICI
from openpilot.common.swaglog import cloudlog
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets.nav_widget import NavWidget
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.mici_setup import (NetworkSetupPage, FailedPage, NetworkConnectivityMonitor,
                                            GreyBigButton, BigPillButton)


class UpdaterNetworkSetupPage(NetworkSetupPage):
  def __init__(self, network_monitor, continue_callback):
    super().__init__(network_monitor, continue_callback, back_callback=None)
    self._continue_button.set_text("download\n& install")
    self._continue_button.set_green(False)


class ProgressPage(NavWidget):
  def __init__(self):
    super().__init__()

    self._progress_title_label = UnifiedLabel("", 64, text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
                                              font_weight=FontWeight.DISPLAY, line_height=0.8)
    self._progress_percent_label = UnifiedLabel("", 132, text_color=rl.Color(255, 255, 255, int(255 * 0.9 * 0.65)),
                                                font_weight=FontWeight.ROMAN,
                                                alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_BOTTOM)

  def _back_enabled(self) -> bool:
    return False

  def set_progress(self, text: str, value: int):
    self._progress_title_label.set_text(text.replace("_", "_\n") + "...")
    self._progress_percent_label.set_text(f"{value}%")

  def show_event(self):
    super().show_event()
    self._nav_bar._alpha = 0.0  # not dismissable
    self.set_progress("downloading", 0)

  def _render(self, rect: rl.Rectangle):
    rl.draw_rectangle_rec(rect, rl.BLACK)
    self._progress_title_label.render(rl.Rectangle(
      rect.x + 12,
      rect.y + 2,
      rect.width,
      self._progress_title_label.get_content_height(int(rect.width - 20)),
    ))

    self._progress_percent_label.render(rl.Rectangle(
      rect.x + 12,
      rect.y + 18,
      rect.width,
      rect.height,
    ))


class Updater(Scroller):
  def __init__(self, updater_path, manifest_path):
    super().__init__()
    self.updater = updater_path
    self.manifest = manifest_path

    self.progress_value = 0
    self.progress_text = "loading"
    self.process = None
    self.update_thread = None
    self._update_failed = False

    self._network_monitor = NetworkConnectivityMonitor()
    self._network_monitor.start()

    self._network_setup_page = UpdaterNetworkSetupPage(self._network_monitor, self._network_setup_continue_callback)

    self._progress_page = ProgressPage()

    self._failed_page = FailedPage(self._retry, title="update failed")

    self._continue_button = BigPillButton("next")
    self._continue_button.set_click_callback(lambda: gui_app.push_widget(self._network_setup_page))

    self._scroller.add_widgets([
      GreyBigButton("update required", "the download size\nis approximately 1 GB",
                    gui_app.texture("icons_mici/offroad_alerts/green_wheel.png", 64, 64)),
      self._continue_button,
    ])

    gui_app.add_nav_stack_tick(self._nav_stack_tick)

  def _network_setup_continue_callback(self, _):
    self.install_update()

  def _retry(self):
    gui_app.pop_widgets_to(self)

  def _nav_stack_tick(self):
    self._progress_page.set_progress(self.progress_text, self.progress_value)

    if self._update_failed:
      self._update_failed = False
      self.show_event()
      gui_app.pop_widgets_to(self, lambda: gui_app.push_widget(self._failed_page))

  def install_update(self):
    self.progress_value = 0
    self.progress_text = "downloading"

    def start_update():
      self.update_thread = threading.Thread(target=self._run_update_process, daemon=True)
      self.update_thread.start()

    # Start the update process in a separate thread *after* show animation completes
    self._progress_page.set_shown_callback(start_update)
    gui_app.push_widget(self._progress_page)

  def _run_update_process(self):
    # TODO: just import it and run in a thread without a subprocess
    try:
      cmd = [self.updater, "--swap", self.manifest]
      self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      text=True, bufsize=1, universal_newlines=True)
    except Exception:
      self._update_failed = True
      return

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
      self._update_failed = True

  def close(self):
    self._network_monitor.stop()


def main():
  config_realtime_process(0, 51)
  # attempt to affine. AGNOS will start setup with all cores, should only fail when manually launching with screen off
  if TICI:
    try:
      set_core_affinity([5])
    except OSError:
      cloudlog.exception("Failed to set core affinity for updater process")

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
