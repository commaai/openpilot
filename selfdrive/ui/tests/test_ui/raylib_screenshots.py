#!/usr/bin/env python3
import os
import sys
import shutil
import time
import pathlib
from collections import namedtuple
from collections.abc import Callable
from typing import NotRequired, TypedDict

import pyautogui
import pywinctl
from PIL import ImageChops

from cereal import log
from cereal import messaging
from cereal.messaging import PubMaster
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.test.helpers import with_processes
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from openpilot.system.updated.updated import parse_release_notes

TEST_DIR = pathlib.Path(__file__).parent
TEST_OUTPUT_DIR = TEST_DIR / "raylib_report"
SCREENSHOTS_DIR = TEST_OUTPUT_DIR / "screenshots"
UI_DELAY = 0.2
SCROLL_DELAY = 1.5  # Delay screenshot by this many seconds after scrolling (to allow scroll to settle)
DEFAULT_SCROLL_AMOUNT = -20  # Good for most full screen scrollers
MAX_SCREENSHOTS_PER_CASE = 8  # Maximum screenshots to generate while scrolling


# Offroad alerts to test
OFFROAD_ALERTS = ['Offroad_IsTakingSnapshot']


def put_update_params(params: Params):
  params.put("UpdaterCurrentReleaseNotes", parse_release_notes(BASEDIR))
  params.put("UpdaterNewReleaseNotes", parse_release_notes(BASEDIR))
  description = "0.10.1 / this-is-a-really-super-mega-long-branch-name / 7864838 / Oct 03"
  params.put("UpdaterCurrentDescription", description)
  params.put("UpdaterNewDescription", description)


def setup_homescreen(click, pm: PubMaster):
  pass


def setup_settings(click, pm: PubMaster):
  click(100, 100)


def close_settings(click, pm: PubMaster):
  click(240, 216)


def setup_settings_network(click, pm: PubMaster):
  setup_settings(click, pm)
  click(278, 450)


def setup_settings_toggles(click, pm: PubMaster):
  setup_settings(click, pm)
  click(278, 600)


def setup_settings_software(click, pm: PubMaster):
  put_update_params(Params())
  setup_settings(click, pm)
  click(278, 720)


def setup_settings_firehose(click, pm: PubMaster):
  setup_settings(click, pm)
  click(278, 845)


def setup_settings_developer(click, pm: PubMaster):
  setup_settings(click, pm)
  click(278, 950)


def setup_keyboard(click, pm: PubMaster):
  setup_settings_developer(click, pm)
  click(1930, 470)


def setup_pair_device(click, pm: PubMaster):
  click(1950, 800)


def setup_offroad_alert(click, pm: PubMaster):
  set_offroad_alert("Offroad_TemperatureTooHigh", True, extra_text='99C')
  set_offroad_alert("Offroad_ExcessiveActuation", True, extra_text='longitudinal')
  for alert in OFFROAD_ALERTS:
    set_offroad_alert(alert, True)

  setup_settings(click, pm)
  close_settings(click, pm)


def setup_confirmation_dialog(click, pm: PubMaster):
  setup_settings(click, pm)
  click(1985, 791)  # reset calibration


def setup_homescreen_update_available(click, pm: PubMaster):
  params = Params()
  params.put_bool("UpdateAvailable", True)
  put_update_params(params)
  setup_settings(click, pm)
  close_settings(click, pm)


def setup_software_release_notes(click, pm: PubMaster):
  setup_settings(click, pm)
  setup_settings_software(click, pm)
  click(588, 110)  # expand description for current version


class CaseConfig(TypedDict):
  scroll_amount: NotRequired[int]
  scroll_enabled: NotRequired[bool]


SetupFunction = Callable[[Callable[..., None], PubMaster], None]
CaseValue = SetupFunction | tuple[SetupFunction, CaseConfig | None]

# Value can be the setup function, or tuple of (setup func, config)
CASES: dict[str, CaseValue] = {
  "homescreen": setup_homescreen,
  "settings_device": setup_settings,
  "settings_network": setup_settings_network,
  "settings_toggles": setup_settings_toggles,
  "settings_software": setup_settings_software,
  "settings_firehose": setup_settings_firehose,
  "settings_developer": setup_settings_developer,
  "keyboard": (setup_keyboard, {"scroll_enabled": False}),  # The blinking cursor makes it think there was a change when scrolling
  "pair_device": setup_pair_device,
  "offroad_alert": (setup_offroad_alert, {"scroll_amount": -12}),  # smaller scrollable area
  "homescreen_update_available": (setup_homescreen_update_available, {"scroll_amount": -12}),  # smaller scrollable area
  "confirmation_dialog": setup_confirmation_dialog,
  "software_release_notes": setup_software_release_notes,
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

  def screenshot(self):
    full_screenshot = pyautogui.screenshot()
    cropped = full_screenshot.crop((self.ui.left, self.ui.top, self.ui.left + self.ui.width, self.ui.top + self.ui.height))
    return cropped

  def screenshot_and_save(self, name: str):
    screenshot = self.screenshot()
    screenshot.save(SCREENSHOTS_DIR / f"{name}.png")
    return screenshot

  def capture_scrollable(self, name: str, scroll_clicks: int, max_screenshots=MAX_SCREENSHOTS_PER_CASE):
    # Take first screenshot
    prev = self.screenshot_and_save(name)

    # Scroll until there are no more changes or we reach the limit
    for i in range(1, max_screenshots):
      self.scroll(scroll_clicks)
      time.sleep(SCROLL_DELAY)
      curr = self.screenshot()

      # Check for difference
      try:
        # TODO: This might need to be more robust to allow for small pixel diffs in case scrolling isn't consistent, but so far it seems to work
        diff = ImageChops.difference(prev.convert('RGB'), curr.convert('RGB'))
        if diff.getbbox() is None:
          # no changes -> reached end
          break
      except Exception as e:
        print(f"error comparing screenshots: {e}")
        break

      # Save the current page
      curr.save(SCREENSHOTS_DIR / f"{name}_{i}.png")

      prev = curr

  def click(self, x: int, y: int, *args, **kwargs):
    pyautogui.mouseDown(self.ui.left + x, self.ui.top + y, *args, **kwargs)
    time.sleep(0.01)
    pyautogui.mouseUp(self.ui.left + x, self.ui.top + y, *args, **kwargs)

  def scroll(self, clicks: int, *args, **kwargs):
    if clicks == 0:
      return
    click = -1 if clicks < 0 else 1  # -1 = down, 1 = up
    for _ in range(abs(clicks)):
      pyautogui.scroll(click, *args, **kwargs)  # scroll for individual clicks since we need to delay between clicks
      time.sleep(0.01)  # small delay between scroll clicks to work properly

  @with_processes(["ui"])
  def test_ui(self, name: str, setup_case: SetupFunction, config: CaseConfig | None = None):
    self.setup()
    time.sleep(UI_DELAY)  # Wait for UI to start
    setup_case(self.click, self.pm)
    config = config or {}

    # Just take a screenshot if scrolling is disabled
    scroll_enabled = config.get("scroll_enabled", True)
    if not scroll_enabled:
      self.screenshot_and_save(name)
      return

    try:
      scroll_clicks = config.get("scroll_amount", DEFAULT_SCROLL_AMOUNT)
      self.capture_scrollable(name, scroll_clicks=scroll_clicks)
    except Exception as e:
      print(f"failed capturing scrollable page, falling back to single screenshot: {e}")
      self.screenshot_and_save(name)


def create_screenshots():
  if TEST_OUTPUT_DIR.exists():
    shutil.rmtree(TEST_OUTPUT_DIR)
  SCREENSHOTS_DIR.mkdir(parents=True)

  t = TestUI()
  with OpenpilotPrefix():
    params = Params()
    params.put("DongleId", "123456789012345")
    for name, setup in CASES.items():
      setup_fn, cfg = setup if isinstance(setup, tuple) else (setup, None)
      t.test_ui(name, setup_fn, cfg)


if __name__ == "__main__":
  create_screenshots()
