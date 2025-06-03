#!/usr/bin/env python3
import pyray as rl
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.main import MainLayout
from openpilot.selfdrive.ui.ui_state import ui_state


def main():
  gui_app.init_window("UI")
  main_layout = MainLayout()
  for _ in gui_app.render():
    ui_state.update()

    #TODO handle brigntness and awake state here

    main_layout.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))


if __name__ == "__main__":
  main()
