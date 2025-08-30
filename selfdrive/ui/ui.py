#!/usr/bin/env python3
import pyray as rl
from openpilot.common.watchdog import kick_watchdog
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.main import MainLayout
from openpilot.selfdrive.ui.ui_state import ui_state


def main():
  gui_app.init_window("UI")
  main_layout = MainLayout()
  main_layout.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
  for _ in gui_app.render():
    ui_state.update()

    # TODO handle brigntness and awake state here

    main_layout.render()

    kick_watchdog()


if __name__ == "__main__":
  main()
