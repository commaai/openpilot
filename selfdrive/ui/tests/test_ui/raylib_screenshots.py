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


class RaylibUIScreenShooter:
  def __init__(self):
    os.environ.setdefault("SCALE", "1")
    # speed up CI by avoiding mouseinfo import from pyautogui
    import sys
    sys.modules["mouseinfo"] = False

  def _init_pm(self):
    # minimal set for offroad home/settings
    services = [
      "deviceState",
      "selfdriveState",
      "carParams",
    ]
    self.pm = PubMaster(services)

    # seed deviceState for offroad
    ds = log.Event.new_message('deviceState').deviceState
    ds = log.Event.new_message('deviceState').as_builder()
    ds.deviceState.networkType = log.DeviceState.NetworkType.wifi
    ds.deviceState.started = False
    for _ in range(5):
      self.pm.send('deviceState', ds)
      ds.clear_write_flag()
      time.sleep(0.05)

    # default prime/device params
    p = Params()
    p.put("DongleId", "123456789012345")

  def _locate_window(self):
    try:
      self.ui = pywinctl.getWindowsWithTitle("ui")[0]
    except Exception:
      # fallback for headless/Xvfb
      self.ui = namedtuple("bb", ["left", "top", "width", "height"])(0, 0, 2160, 1080)

  def _screenshot(self, name: str):
    im = pyautogui.screenshot(SCREENSHOTS_DIR / f"{name}.png", region=(self.ui.left, self.ui.top, self.ui.width, self.ui.height))
    assert im.width == 2160 and im.height == 1080

  def _click(self, x: int, y: int):
    pyautogui.click(self.ui.left + x, self.ui.top + y)
    time.sleep(UI_DELAY)

  @with_processes(["ui"])
  def create_two(self):
    self._init_pm()
    self._locate_window()

    # homescreen
    self._screenshot("homescreen")

    # settings: mimic existing raylib test behavior (top-left touch opens settings)
    self._click(100, 100)
    self._screenshot("settings_device")


def main():
  if TEST_OUTPUT_DIR.exists():
    shutil.rmtree(TEST_OUTPUT_DIR)
  SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

  with OpenpilotPrefix():
    RaylibUIScreenShooter().create_two()


if __name__ == "__main__":
  main()


