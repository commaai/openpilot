#!/usr/bin/env python3
import os
import time
import coverage
import pyray as rl
from unittest.mock import patch
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
from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout

FPS = 60
HEADLESS = False  # os.getenv("WINDOWED", "0") == "1"
STARTED = False

# patch ui_state to set onroad/offroad based on our script events
# _original_update_state = ui_state._update_state


# @patch('ui_state._update_state', autospec=True)
def _update_state():
  # _original_update_state(self)
  ui_state.started = STARTED


# ui_state.update_state = _update_state.__get__(ui_state, type(ui_state))
ui_state._update_state = _update_state


@dataclass
class DummyEvent:
  click: bool = False
  # TODO: add some kind of intensity
  swipe_left: bool = False
  swipe_right: bool = False
  swipe_down: bool = False
  onroad: bool = False
  offroad: bool = False


SCRIPT = [
  (0, DummyEvent()),
  (FPS * 1, DummyEvent(swipe_left=True, onroad=True)),
  # (FPS * 1, DummyEvent(click=True)),
  # (FPS * 2, DummyEvent(click=True)),
  # (FPS * 3, DummyEvent(swipe_down=True)),
  # (FPS * 4, DummyEvent(swipe_down=True)),
  # (FPS * 5, DummyEvent(swipe_left=True, onroad=True)),
  (FPS * 10, DummyEvent()),
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
  global STARTED
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
  if event.onroad:
    # ui_state.started = True
    STARTED = True
  if event.offroad:
    # ui_state.started = False
    STARTED = False


def run_replay():
  setup_state()
  os.makedirs(DIFF_OUT_DIR, exist_ok=True)

  if HEADLESS:
    rl.set_config_flags(rl.FLAG_WINDOW_HIDDEN)
  gui_app.init_window("ui diff test", fps=FPS)
  main_layout = MiciMainLayout()
  main_layout.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))

  frame = 0
  script_index = 0

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
  cov.stop()
  cov.save()
  cov.report()
  cov.html_report(directory=os.path.join(DIFF_OUT_DIR, 'htmlcov'))
  print("HTML report: htmlcov/index.html")


if __name__ == "__main__":
  main()
