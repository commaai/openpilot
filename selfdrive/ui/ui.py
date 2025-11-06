#!/usr/bin/env python3
import os
import pyray as rl

from openpilot.common.realtime import config_realtime_process, set_core_affinity
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.main import MainLayout
from openpilot.selfdrive.ui.ui_state import ui_state


def main():
  cores = {5, }
  config_realtime_process(0, 51)

  gui_app.init_window("UI")
  main_layout = MainLayout()
  main_layout.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
  for should_render in gui_app.render():
    ui_state.update()
    if should_render:
      main_layout.render()

      # Reaffine after power save offlines our core
      try:
          if hasattr(os, 'sched_getaffinity') and os.sched_getaffinity(0) != cores:
              set_core_affinity(list(cores))
      except (OSError, AttributeError):
          pass  # Affinity setting unavailable or failed

if __name__ == "__main__":
  main()
