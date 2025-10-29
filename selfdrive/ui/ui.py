#!/usr/bin/env python3
import pyray as rl

from openpilot.common.realtime import config_realtime_process
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.main import MainLayout
from openpilot.selfdrive.ui.ui_state import ui_state


def main():
  config_realtime_process([1, 2], 1)

  gui_app.init_window("UI")
  main_layout = MainLayout()
  main_layout.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
  for should_render in gui_app.render():
    ui_state.update()
    if should_render:
      main_layout.render()


if __name__ == "__main__":
  main()
