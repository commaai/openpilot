#!/usr/bin/env python3
import os
import time
import coverage
import pyray as rl
from dataclasses import dataclass
from openpilot.selfdrive.ui.tests.diff.diff import DIFF_OUT_DIR

os.environ["RECORD"] = "1"
if "RECORD_OUTPUT" not in os.environ:
  os.environ["RECORD_OUTPUT"] = "mici_ui_replay.mp4"

os.environ["RECORD_OUTPUT"] = os.path.join(DIFF_OUT_DIR, os.environ["RECORD_OUTPUT"])

from openpilot.common.params import Params
from openpilot.system.version import terms_version, training_version
from openpilot.system.ui.lib.application import gui_app, MousePos, MouseEvent
from openpilot.selfdrive.ui.ui_state import ui_state

FPS = 60
HEADLESS = os.getenv("WINDOWED", "0") == "1"


@dataclass
class DummyEvent:
  click: bool = False
  # TODO: add some kind of intensity
  swipe_left: bool = False
  swipe_right: bool = False
  swipe_down: bool = False


SCRIPT = [
  (0, DummyEvent()),
  (FPS * 1, DummyEvent(click=True)),
  (FPS * 2, DummyEvent(click=True)),
  (FPS * 3, DummyEvent()),
]


def setup_state():
  params = Params()
  params.put("HasAcceptedTerms", terms_version)
  params.put("CompletedTrainingVersion", training_version)
  params.put("DongleId", "test123456789")
  params.put("UpdaterCurrentDescription", "0.10.1 / test-branch / abc1234 / Nov 30")
  return None


def inject_click(coords):
  events = []
  x, y = coords[0]
  events.append(MouseEvent(pos=MousePos(x, y), slot=0, left_pressed=True, left_released=False, left_down=False, t=time.monotonic()))
  for x, y in coords[1:]:
    events.append(MouseEvent(pos=MousePos(x, y), slot=0, left_pressed=False, left_released=False, left_down=True, t=time.monotonic()))
  x, y = coords[-1]
  events.append(MouseEvent(pos=MousePos(x, y), slot=0, left_pressed=False, left_released=True, left_down=False, t=time.monotonic()))

  with gui_app._mouse._lock:
    gui_app._mouse._events.extend(events)


def handle_event(event: DummyEvent):
  if event.click:
    inject_click([(gui_app.width // 2, gui_app.height // 2)])
  if event.swipe_left:
    inject_click([(gui_app.width * 3 // 4, gui_app.height // 2),
                  (gui_app.width // 4, gui_app.height // 2),
                  (0, gui_app.height // 2)])
  if event.swipe_right:
    inject_click([(gui_app.width // 4, gui_app.height // 2),
                  (gui_app.width * 3 // 4, gui_app.height // 2),
                  (gui_app.width, gui_app.height // 2)])
  if event.swipe_down:
    inject_click([(gui_app.width // 2, gui_app.height // 4),
                  (gui_app.width // 2, gui_app.height * 3 // 4),
                  (gui_app.width // 2, gui_app.height)])


def run_replay():
  setup_state()
  os.makedirs(DIFF_OUT_DIR, exist_ok=True)

  if not HEADLESS:
    rl.set_config_flags(rl.FLAG_WINDOW_HIDDEN)
  gui_app.init_window("ui diff test", fps=FPS)

  from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout  # import here for coverage

  main_layout = MiciMainLayout()
  main_layout.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))

  script_index = 0
  frame = 0
  # Override raylib timing functions to return deterministic values based on frame count instead of real time
  rl.get_frame_time = lambda: 1.0 / FPS
  rl.get_time = lambda: frame / FPS

  for should_render in gui_app.render():
    while script_index < len(SCRIPT) and SCRIPT[script_index][0] == frame:
      _, event = SCRIPT[script_index]
      handle_event(event)
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
  cov.save()
  cov.report()
  cov.html_report(directory=os.path.join(DIFF_OUT_DIR, 'htmlcov'))
  print("HTML report: htmlcov/index.html")


if __name__ == "__main__":
  main()
