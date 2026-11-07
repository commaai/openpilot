#!/usr/bin/env python3
import os
import time

from openpilot.cereal import messaging
from openpilot.common.hardware import COMMA_HARDWARE, HARDWARE
from openpilot.common.realtime import Priority, config_realtime_process, set_core_affinity
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.main import MainLayout
from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout
from openpilot.selfdrive.ui.ui_state import ui_state

BIG_UI = gui_app.big_ui()

FRAME_CLOCK_TIMEOUT_MS = 50  # one camera period


class DriverMonitoringClock:
  """Presents onroad frames just after driver monitoring, the last GPU job in each camera period."""
  def __init__(self):
    self._dm = messaging.sub_sock('driverStateV2', conflate=True, timeout=FRAME_CLOCK_TIMEOUT_MS)
    self._dm_running = False

  def wait(self) -> bool:
    if not ui_state.started:
      return False
    if self._dm.receive(non_blocking=True) is not None:
      self._dm_running = True
      return True
    # without DM keep the fixed rate rather than waiting out a timeout every frame
    if self._dm_running:
      self._dm_running = self._dm.receive() is not None
    return self._dm_running


def main():
  cores = {5, }
  # above plannerd and radard
  config_realtime_process(0, Priority.CTRL_HIGH)

  gui_app.init_window("UI")
  if BIG_UI:
    MainLayout()
  else:
    MiciMainLayout()

  if HARDWARE.get_device_type() == 'tizi':
    gui_app.set_frame_clock(DriverMonitoringClock().wait)

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
