#!/usr/bin/env python3
import os
import pyray as rl
from openpilot.common.watchdog import kick_watchdog
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.main import MainLayout
from openpilot.selfdrive.ui.ui_state import ui_state


def main():
  # Set highest process priority for real-time UI responsiveness
  os.nice(-20)

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
