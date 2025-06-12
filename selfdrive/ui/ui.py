#!/usr/bin/env python3
import pyray as rl
from openpilot.common.watchdog import kick_watchdog
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.main import MainLayout
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.lib.device_state import DeviceState


def main():
  gui_app.init_window("UI")

  device_state = DeviceState(ui_state)
  main_layout = MainLayout(device_state)
  main_layout.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))

  for _ in gui_app.render():
    ui_state.update()
    device_state.update()

    main_layout.render()

    kick_watchdog()


if __name__ == "__main__":
  main()
