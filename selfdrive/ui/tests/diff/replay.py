#!/usr/bin/env python3
import os
import time
import coverage
import pyray as rl
from openpilot.selfdrive.ui.tests.diff.diff import DIFF_OUT_DIR

os.environ["RECORD"] = "1"
if "RECORD_OUTPUT" not in os.environ:
  os.environ["RECORD_OUTPUT"] = "mici_ui_replay.mp4"

os.environ["RECORD_OUTPUT"] = os.path.join(DIFF_OUT_DIR, os.environ["RECORD_OUTPUT"])

from openpilot.common.params import Params
from openpilot.system.version import terms_version, training_version
from openpilot.system.ui.lib.application import gui_app, MousePos, MouseEvent
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout

FPS = 60
HEADLESS = os.getenv("WINDOWED", "0") == "1"

SCRIPT = [
  (0, None),
  (FPS * 1, (100, 100)),
  (FPS * 2, None),
]


def setup_state():
  params = Params()
  params.put("HasAcceptedTerms", terms_version)
  params.put("CompletedTrainingVersion", training_version)
  params.put("DongleId", "test123456789")
  params.put("UpdaterCurrentDescription", "0.10.1 / test-branch / abc1234 / Nov 30")
  return None


def inject_click(x, y):
  press_event = MouseEvent(pos=MousePos(x, y), slot=0, left_pressed=True, left_released=False, left_down=True, t=time.monotonic())

  release_event = MouseEvent(pos=MousePos(x, y), slot=0, left_pressed=False, left_released=True, left_down=False, t=time.monotonic())

  with gui_app._mouse._lock:
    gui_app._mouse._events.append(press_event)
    gui_app._mouse._events.append(release_event)


def run_replay():
  setup_state()
  os.makedirs(DIFF_OUT_DIR, exist_ok=True)

  if not HEADLESS:
    rl.set_config_flags(rl.FLAG_WINDOW_HIDDEN)
  gui_app.init_window("ui diff test", fps=FPS)
  main_layout = MiciMainLayout()
  main_layout.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))

  frame = 0
  script_index = 0

  for should_render in gui_app.render():
    while script_index < len(SCRIPT) and SCRIPT[script_index][0] == frame:
      _, coords = SCRIPT[script_index]
      if coords is not None:
        inject_click(*coords)
      script_index += 1

    ui_state.update()

    if should_render:
      main_layout.render()

    frame += 1

    if script_index >= len(SCRIPT):
      break

  gui_app.close()

  print(f"Total frames: {frame}")
  print(f"Video saved to: {os.environ['RECORD_OUTPUT']}")


def main():
  cov = coverage.coverage(source=['openpilot.selfdrive.ui.mici'])
  with cov.collect():
    run_replay()
  cov.stop()
  cov.save()
  cov.report()
  cov.html_report(directory='htmlcov')
  print("HTML report: htmlcov/index.html")


if __name__ == "__main__":
  main()
