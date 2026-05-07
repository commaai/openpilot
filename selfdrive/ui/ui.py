#!/usr/bin/env python3
import os
import time

from cereal import messaging
from openpilot.system.hardware import TICI
from openpilot.common.realtime import config_realtime_process, set_core_affinity
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.main import MainLayout
from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout
from openpilot.selfdrive.ui.ui_state import ui_state

BIG_UI = gui_app.big_ui()


def main():
  cores = {5, }
  config_realtime_process(list(cores), 99)

  gui_app.init_window("UI")
  if BIG_UI:
    MainLayout()
  else:
    MiciMainLayout()

  pm = messaging.PubMaster(['uiDebug'])
  for should_render, frame_time, cpu_time in gui_app.render():
    t_yield_start = time.monotonic()
    ui_state.update()
    ui_state_update_time = time.monotonic() - t_yield_start

    if should_render:
      # reaffine after power save offlines our core
      t_reaffine_start = time.monotonic()
      if TICI and os.sched_getaffinity(0) != cores:
        try:
          set_core_affinity(list(cores))
        except OSError:
          pass
      reaffine_time = time.monotonic() - t_reaffine_start

      msg = messaging.new_message('uiDebug')
      msg.uiDebug.cpuTimeMillis = (cpu_time + ui_state_update_time + reaffine_time) * 1000
      msg.uiDebug.frameTimeMillis = frame_time * 1000
      msg.uiDebug.uiStateUpdateMillis = ui_state_update_time * 1000
      msg.uiDebug.reaffineMillis = reaffine_time * 1000
      pm.send('uiDebug', msg)


if __name__ == "__main__":
  main()
