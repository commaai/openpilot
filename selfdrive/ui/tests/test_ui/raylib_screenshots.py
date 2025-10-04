#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
import time
import pathlib
from collections import namedtuple

import pyautogui
import pywinctl

from cereal import log
from cereal import messaging
from cereal.messaging import PubMaster
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.test.helpers import with_processes, processes_context
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert

TEST_DIR = pathlib.Path(__file__).parent
TEST_OUTPUT_DIR = TEST_DIR / "raylib_report"
SCREENSHOTS_DIR = TEST_OUTPUT_DIR / "screenshots"
UI_DELAY = 0.1

# Offroad alerts to test
OFFROAD_ALERTS = ['Offroad_IsTakingSnapshot']


def setup_homescreen(click, pm: PubMaster):
  pass


def setup_settings_device(click, pm: PubMaster):
  click(100, 100)


def setup_settings_network(click, pm: PubMaster):
  setup_settings_device(click, pm)
  click(278, 450)


def setup_settings_toggles(click, pm: PubMaster):
  setup_settings_device(click, pm)
  click(278, 600)


def setup_settings_software(click, pm: PubMaster):
  setup_settings_device(click, pm)
  click(278, 720)


def setup_settings_firehose(click, pm: PubMaster):
  setup_settings_device(click, pm)
  click(278, 845)


def setup_settings_developer(click, pm: PubMaster):
  setup_settings_device(click, pm)
  click(278, 950)


def setup_keyboard(click, pm: PubMaster):
  setup_settings_developer(click, pm)
  click(1930, 270)


def setup_pair_device(click, pm: PubMaster):
  click(1950, 800)


def setup_offroad_alert(click, pm: PubMaster):
  set_offroad_alert("Offroad_TemperatureTooHigh", True, extra_text='99C')
  set_offroad_alert("Offroad_ExcessiveActuation", True, extra_text='longitudinal')
  for alert in OFFROAD_ALERTS:
    set_offroad_alert(alert, True)

  setup_settings_device(click, pm)
  click(240, 216)


CASES = {
  "homescreen": setup_homescreen,
  "settings_device": setup_settings_device,
  "settings_network": setup_settings_network,
  "settings_toggles": setup_settings_toggles,
  "settings_software": setup_settings_software,
  "settings_firehose": setup_settings_firehose,
  "settings_developer": setup_settings_developer,
  "keyboard": setup_keyboard,
  "pair_device": setup_pair_device,
  "offroad_alert": setup_offroad_alert,
}


class TestUI:
  def __init__(self):
    os.environ["SCALE"] = os.getenv("SCALE", "1")
    sys.modules["mouseinfo"] = False

  def setup(self):
    # Seed minimal offroad state
    self.pm = PubMaster(["deviceState"])
    ds = messaging.new_message('deviceState')
    ds.deviceState.networkType = log.DeviceState.NetworkType.wifi
    for _ in range(5):
      self.pm.send('deviceState', ds)
      ds.clear_write_flag()
      time.sleep(0.05)
    time.sleep(0.5)
    try:
      self.ui = pywinctl.getWindowsWithTitle("UI")[0]
    except Exception as e:
      print(f"failed to find ui window, assuming that it's in the top left (for Xvfb) {e}")
      self.ui = namedtuple("bb", ["left", "top", "width", "height"])(0, 0, 2160, 1080)

  def screenshot(self, name: str):
    full_screenshot = pyautogui.screenshot()
    cropped = full_screenshot.crop((self.ui.left, self.ui.top, self.ui.left + self.ui.width, self.ui.top + self.ui.height))
    cropped.save(SCREENSHOTS_DIR / f"{name}.png")

  def click(self, x: int, y: int, *args, **kwargs):
    pyautogui.mouseDown(self.ui.left + x, self.ui.top + y, *args, **kwargs)
    time.sleep(0.01)
    pyautogui.mouseUp(self.ui.left + x, self.ui.top + y, *args, **kwargs)

  @with_processes(["raylib_ui"])
  def test_ui(self, name, setup_case):
    self.setup()
    setup_case(self.click, self.pm)
    self.screenshot(name)


def create_screenshots():
  if TEST_OUTPUT_DIR.exists():
    shutil.rmtree(TEST_OUTPUT_DIR)
  SCREENSHOTS_DIR.mkdir(parents=True)

  t = TestUI()
  with OpenpilotPrefix():
    params = Params()
    params.put("DongleId", "123456789012345")
    for name, setup in CASES.items():
      t.test_ui(name, setup)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--add", type=str, help="Add a new case with the given name")
  args = parser.parse_args()

  if args.add:
    # Interactive mode: start the UI and wait until it is closed
    name = args.add
    print(f"[add] Starting raylib_ui for interactive case '{name}'. Close the UI window to exit.")

    # Ensure output dir exists
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Seed minimal offroad state
    pm = PubMaster(["deviceState"])
    ds = messaging.new_message('deviceState')
    ds.deviceState.networkType = log.DeviceState.NetworkType.wifi
    for _ in range(5):
      pm.send('deviceState', ds)
      ds.clear_write_flag()
      time.sleep(0.05)

    # Enable click logging to a file under the report directory
    click_log_path = TEST_OUTPUT_DIR / f"clicks_{name}.csv"
    os.environ["CLICK_LOG"] = str(click_log_path)

    with processes_context(["raylib_ui"], init_time=0):
      # Try to locate the UI window; if not found, assume Xvfb top-left position
      try:
        ui = pywinctl.getWindowsWithTitle("UI")[0]
        print(f"[add] UI window found at ({ui.left}, {ui.top}) size {ui.width}x{ui.height}")
      except Exception as e:
        print(f"[add] Failed to find UI window, will poll until it appears: {e}")
        ui = None

      # Poll until the UI window appears
      start_time = time.monotonic()
      while ui is None and (time.monotonic() - start_time) < 10.0:
        try:
          ui = pywinctl.getWindowsWithTitle("UI")[0]
        except Exception:
          ui = None
        time.sleep(UI_DELAY)

      if ui is None:
        print("[add] UI window not found; running until process exits.")

      # Main loop: wait for the UI window to close (or process to exit via context manager)
      try:
        while True:
          # If we can see the window, check it's still valid; otherwise just idle
          if ui is not None:
            try:
              # Access a property to ensure window handle is still valid
              _ = ui.title
            except Exception:
              print("[add] UI window closed.")
              break
          time.sleep(UI_DELAY)
      except KeyboardInterrupt:
        print("[add] Interrupted, stopping.")

    if click_log_path.exists():
      print(f"[add] Clicks stored at: {click_log_path}")
    else:
      print("[add] No clicks recorded or file missing.")

  else:
    create_screenshots()
