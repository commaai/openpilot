#!/usr/bin/env python3
import pyray as rl
from openpilot.common.realtime import config_realtime_process
from openpilot.common.watchdog import kick_watchdog
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.main import MainLayout
from openpilot.selfdrive.ui.ui_state import ui_state


def main():
  # Configure real-time process for optimal UI performance
  # Use real-time scheduling with high priority
  # Core 0 is shared with other non-critical processes (calibrationd, locationd, etc.)
  # Priority 54 for consistent UI performance
  config_realtime_process(0, 54)

  gui_app.init_window("UI")
  main_layout = MainLayout()
  main_layout.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
  for showing_dialog in gui_app.render():
    ui_state.update()

    kick_watchdog()

    if not showing_dialog:
      main_layout.render()


if __name__ == "__main__":
  main()
