#!/usr/bin/env python3
import os
import sys
import threading
import time

from openpilot.cereal import messaging
from openpilot.common.hardware import COMMA_HARDWARE
from openpilot.common.realtime import Priority, config_realtime_process
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.main import MainLayout
from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout
from openpilot.selfdrive.ui.ui_state import ui_state

BIG_UI = gui_app.big_ui()


def set_core_affinity(cores: set[int]):
  # every thread follows the renderer: they take the GIL it needs, and on its core they only run while it waits for vsync.
  # on another core, that core's realtime processes can preempt one holding the GIL, stalling the renderer for a frame
  os.sched_setaffinity(0, cores)
  for thread in threading.enumerate():
    try:
      if thread.native_id is not None:  # None until it's started
        os.sched_setaffinity(thread.native_id, cores)
    except ProcessLookupError:  # it's already gone
      pass


def main():
  cores = {5, }
  # above plannerd and radard. the threads it starts don't inherit that, so they can't hold off the renderer
  config_realtime_process(0, Priority.CTRL_HIGH)
  if COMMA_HARDWARE:
    os.sched_setscheduler(0, os.SCHED_FIFO | os.SCHED_RESET_ON_FORK, os.sched_param(Priority.CTRL_HIGH))
  # a thread waiting for the GIL takes it from the one holding it after this long
  sys.setswitchinterval(0.001)

  gui_app.init_window("UI")
  if BIG_UI:
    MainLayout()
  else:
    MiciMainLayout()

  pm = messaging.PubMaster(['uiDebug'])
  for should_render, frame_time, cpu_time in gui_app.render():
    extra_start = time.monotonic()
    ui_state.update()

    if should_render:
      # reaffine after power save offlines our core
      if COMMA_HARDWARE and os.sched_getaffinity(0) != cores:
        try:
          set_core_affinity(cores)
        except OSError:
          pass

      extra_cpu = time.monotonic() - extra_start
      msg = messaging.new_message('uiDebug')
      msg.uiDebug.cpuTimeMillis = (cpu_time + extra_cpu) * 1000
      msg.uiDebug.frameTimeMillis = frame_time * 1000
      pm.send('uiDebug', msg)


if __name__ == "__main__":
  main()
