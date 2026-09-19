#!/usr/bin/env python3
import os
import time

from openpilot.cereal import messaging
from openpilot.common.hardware import COMMA_HARDWARE
from openpilot.common.realtime import Priority, config_realtime_process, set_core_affinity
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.ui_state import ui_state

BIG_UI = gui_app.big_ui()



def boot_stage(stage):
  with open('/tmp/boot-stages', 'a') as f:
    f.write(f'{time.monotonic():.6f} {stage}\n')

def main():
  boot_stage("ui_imported")
  cores = {5, }
  # above plannerd and radard
  config_realtime_process(5 if COMMA_HARDWARE else 0, Priority.CTRL_HIGH)

  gui_app.init_window("UI")
  boot_stage("ui_window_ready")
  if BIG_UI:
    from openpilot.selfdrive.ui.layouts.main import MainLayout
    MainLayout()
  else:
    from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout
    MiciMainLayout()

  boot_stage('ui_layout_ready')
  pm = messaging.PubMaster(['uiDebug'])
  for should_render, frame_time, cpu_time in gui_app.render():
    extra_start = time.monotonic()
    ui_state.update()

    if should_render:
      # reaffine after power save offlines our core
      if COMMA_HARDWARE and os.sched_getaffinity(0) != cores:
        try:
          set_core_affinity(list(cores))
        except OSError:
          pass

      extra_cpu = time.monotonic() - extra_start
      msg = messaging.new_message('uiDebug')
      msg.uiDebug.cpuTimeMillis = (cpu_time + extra_cpu) * 1000
      msg.uiDebug.frameTimeMillis = frame_time * 1000
      pm.send('uiDebug', msg)


if __name__ == "__main__":
  main()
