#!/usr/bin/env python3
import os
import shutil
import time
import pathlib
from collections import namedtuple

import pyautogui
import pywinctl

from cereal import log
from cereal.messaging import PubMaster
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.test.helpers import with_processes


TEST_DIR = pathlib.Path(__file__).parent
TEST_OUTPUT_DIR = TEST_DIR / "report_1"
SCREENSHOTS_DIR = TEST_OUTPUT_DIR / "screenshots"
UI_DELAY = 0.1


def setup_homescreen(click, pm: PubMaster):
  pass


def setup_settings_device(click, pm: PubMaster):
  # open settings from home (top-left click)
  click(100, 100)


CASES = {
  "homescreen": setup_homescreen,
  "settings_device": setup_settings_device,
}


class TestUI:
  def __init__(self):
    os.environ["SCALE"] = os.getenv("SCALE", "1")
    import sys
    sys.modules["mouseinfo"] = False

  def setup(self):
    # Seed minimal offroad state
    self.pm = PubMaster(["deviceState"])
    ds = log.DeviceState.new_message()
    ds.networkType = log.DeviceState.NetworkType.wifi
    ds.started = False
    for _ in range(5):
      self.pm.send('deviceState', ds)
      ds.clear_write_flag()
      time.sleep(0.05)

    # Find the raylib UI window
    time.sleep(1)  # Wait for UI to appear
    ui_windows = pywinctl.getWindowsWithTitle("ui")
    if ui_windows:
      self.ui = ui_windows[0]
    else:
      # Fallback to default dimensions
      self.ui = namedtuple("bb", ["left", "top", "width", "height"])(0, 0, 2160, 1080)

  def screenshot(self, name: str):
    # Take full screenshot and crop to UI window (handles multi-monitor setups)
    full_screenshot = pyautogui.screenshot()
    cropped = full_screenshot.crop((self.ui.left, self.ui.top,
                                  self.ui.left + self.ui.width,
                                  self.ui.top + self.ui.height))
    cropped.save(SCREENSHOTS_DIR / f"{name}.png")

  def click(self, x: int, y: int, *args, **kwargs):
    pyautogui.click(self.ui.left + x, self.ui.top + y, *args, **kwargs)
    time.sleep(UI_DELAY)

  @with_processes(["ui"])  # same decorator pattern
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
  create_screenshots()


