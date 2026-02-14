#!/usr/bin/env python3
import os
import coverage
import pyray as rl
import argparse

from collections.abc import Callable
from typing import Literal

from cereal.messaging import PubMaster
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.ui.tests.diff.diff import DIFF_OUT_DIR
from openpilot.selfdrive.ui.tests.diff.replay_setup import initialize_params

LayoutVariant = Literal["mici", "tizi"]

HEADLESS = os.getenv("WINDOWED", "0") != "1"
FPS = 60


class ReplayContext:
  send_fn: Callable | None = None  # Function to call each frame to send messages for persistent states (onroad, alerts)

  def __init__(self, pm: PubMaster, main_layout) -> None:
    self.pm = pm
    self.send_fn = None
    self.main_layout = main_layout

  def update_send_fn(self, send_fn: Callable) -> None:
    self.send_fn = send_fn
    send_fn() # Call immediately to set initial state

  def get_send_fn(self) -> Callable | None:
    return self.send_fn


def run_replay(variant: LayoutVariant) -> None:
  from openpilot.selfdrive.ui.ui_state import ui_state  # Import within OpenpilotPrefix context so param values are setup correctly
  from openpilot.system.ui.lib.application import gui_app  # Import here for accurate coverage
  from openpilot.selfdrive.ui.tests.diff.replay_script import build_script

  os.makedirs(DIFF_OUT_DIR, exist_ok=True)

  initialize_params()

  if HEADLESS:
    rl.set_config_flags(rl.FLAG_WINDOW_HIDDEN)
  gui_app.init_window("ui diff test", fps=FPS)

  # Dynamically import main layout based on variant
  if variant == "mici":
    from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout as MainLayout
  else:
    from openpilot.selfdrive.ui.layouts.main import MainLayout
  main_layout = MainLayout()
  main_layout.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))

  # Create context and build script
  pm = PubMaster(["deviceState", "pandaStates", "driverStateV2", "selfdriveState"])
  context = ReplayContext(pm, main_layout)
  script = build_script(context, variant)
  script_index = 0

  frame = 0
  # Override raylib timing functions to return deterministic values based on frame count instead of real time
  rl.get_frame_time = lambda: 1.0 / FPS
  rl.get_time = lambda: frame / FPS

  # Main loop to replay events and render frames
  for should_render in gui_app.render():
    # Handle all events for the current frame
    while script_index < len(script) and script[script_index][0] == frame:
      _, event = script[script_index]
      # Call setup function, if any
      if event.setup:
        event.setup()
      # Send mouse events to the application
      if event.mouse_events:
        with gui_app._mouse._lock:
          gui_app._mouse._events.extend(event.mouse_events)
      # Move to next script event
      script_index += 1

    # Keep sending cereal messages for persistent states (onroad, alerts)
    send_fn = context.get_send_fn()
    if send_fn:
      send_fn()

    ui_state.update()

    if should_render:
      main_layout.render()

    frame += 1

    if script_index >= len(script):
      break

  gui_app.close()

  print(f"Total frames: {frame}")
  print(f"Video saved to: {os.environ['RECORD_OUTPUT']}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--big', action='store_true', help='Use big UI layout (tizi/tici) instead of mici layout')
  args = parser.parse_args()

  variant: LayoutVariant = 'tizi' if args.big else 'mici'

  if args.big:
    os.environ["BIG"] = "1"
  os.environ["RECORD"] = "1"
  os.environ["RECORD_OUTPUT"] = os.path.join(DIFF_OUT_DIR, os.environ.get("RECORD_OUTPUT", f"{variant}_ui_replay.mp4"))

  print(f"Running {variant} UI replay...")
  with OpenpilotPrefix():
    sources = ["openpilot.system.ui"]
    if variant == "mici":
      sources.append("openpilot.selfdrive.ui.mici")
      omit = ["**/*tizi*", "**/*tici*"]  # exclude files containing "tizi" or "tici"
    else:
      sources.extend(["openpilot.selfdrive.ui.layouts", "openpilot.selfdrive.ui.onroad", "openpilot.selfdrive.ui.widgets"])
      omit = ["**/*mici*"]  # exclude files containing "mici"
    cov = coverage.Coverage(source=sources, omit=omit)
    with cov.collect():
      run_replay(variant)
    cov.save()
    cov.report()
    directory = os.path.join(DIFF_OUT_DIR, f"htmlcov-{variant}")
    cov.html_report(directory=directory)
    print(f"HTML report: {directory}/index.html")


if __name__ == "__main__":
  main()
