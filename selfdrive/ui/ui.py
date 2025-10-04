#!/usr/bin/env python3
import time
import numpy as np

import pyray as rl
from openpilot.common.watchdog import kick_watchdog
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.main import MainLayout
from openpilot.selfdrive.ui.ui_state import ui_state

frame_times = []

i = 0


def print_data():
  _1percent_high = np.percentile(frame_times, 99) * 1000
  _1percent_low_fps = 1 / np.percentile(frame_times, 99)
  average = np.mean(frame_times) * 1000
  median = np.median(frame_times) * 1000
  stddev = np.std(frame_times) * 1000
  _min = np.min(frame_times) * 1000
  _max = np.max(frame_times) * 1000
  print(f"\nUI 1% high: {_1percent_high:.2f}ms, avg: {average:.2f}ms, median: {median:.2f}ms, stddev: {stddev:.2f}ms, min: {_min:.2f}ms, max: {_max:.2f}ms, 1% low fps: {_1percent_low_fps:.1f}fps")


def main():
  global i
  # TODO: https://github.com/commaai/agnos-builder/pull/490
  # os.nice(-20)

  gui_app.init_window("UI")
  main_layout = MainLayout()
  main_layout.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
  t = time.monotonic()
  for showing_dialog in gui_app.render():
    t = time.monotonic()
    ui_state.update()

    kick_watchdog()

    if not showing_dialog:
      main_layout.render()
    frame_times.append(time.monotonic() - t)

    i += 1
    if i % 100 == 0:
      print_data()
    print("UI loop time", f'{(frame_times[-1]) * 1000:.3f}ms, theoretical fps: {1 / (frame_times[-1]):.1f}')


if __name__ == "__main__":
  main()
