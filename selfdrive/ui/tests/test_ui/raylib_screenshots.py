#!/usr/bin/env python3
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


def setup_settings_software_download(click, pm: PubMaster):
  params = Params()
  # setup_settings_software but with "DOWNLOAD" button to test long text
  params.put("UpdaterState", "idle")
  params.put_bool("UpdaterFetchAvailable", True)
  setup_settings_software(click, pm)


def setup_settings_software_release_notes(click, pm: PubMaster):
  setup_settings_software(click, pm)
  click(588, 110)  # expand description for current version


def setup_settings_firehose(click, pm: PubMaster):
  setup_settings(click, pm)
  click(278, 845)


def setup_settings_developer(click, pm: PubMaster):
  setup_settings(click, pm)
  click(278, 950)


def setup_keyboard(click, pm: PubMaster):
  setup_settings_developer(click, pm)
  click(1930, 470)


def keyboard_toggle_shift(click):
  click(200, 800)  # shift key


def keyboard_toggle_numbers(click):
  click(200, 900)  # toggle numbers key


def keyboard_click_top_two(click):
  click(140, 430)  # q, Q, 1, [
  click(350, 440)  # w, W, 2, ]


def setup_keyboard_uppercase(click, pm: PubMaster):
  setup_keyboard(click, pm)
  keyboard_toggle_shift(click)


def setup_keyboard_numbers(click, pm: PubMaster):
  setup_keyboard(click, pm)
  keyboard_toggle_numbers(click)


def setup_keyboard_text_and_symbols(click, pm: PubMaster):
  setup_keyboard(click, pm)
  keyboard_toggle_shift(click)  # uppercase
  keyboard_click_top_two(click)  # type "Qw" (shift turns off after first letter)
  keyboard_toggle_numbers(click)  # numbers
  keyboard_click_top_two(click)  # type "12"
  keyboard_toggle_shift(click)  # symbols
  keyboard_click_top_two(click)  # type "[]"


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


def setup_experimental_mode_description(click, pm: PubMaster):
  setup_settings_toggles(click, pm)
  click(1200, 280)  # expand description for experimental mode


CASES = {
  "homescreen": setup_homescreen,
  "settings_device": setup_settings,
  "settings_network": setup_settings_network,
  "settings_toggles": setup_settings_toggles,
  "settings_software": setup_settings_software,
  "settings_software_download": setup_settings_software_download,
  "settings_software_release_notes": setup_settings_software_release_notes,
  "settings_firehose": setup_settings_firehose,
  "settings_developer": setup_settings_developer,
  "keyboard": setup_keyboard,
  "keyboard_uppercase": setup_keyboard_uppercase,
  "keyboard_numbers": setup_keyboard_numbers,
  "keyboard_text_and_symbols": setup_keyboard_text_and_symbols,
  "pair_device": setup_pair_device,
  "offroad_alert": setup_offroad_alert,
  "homescreen_update_available": setup_homescreen_update_available,
  "confirmation_dialog": setup_confirmation_dialog,
  "experimental_mode_description": setup_experimental_mode_description,
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

  @with_processes(["ui"])
  def test_ui(self, name, setup_case):
    self.setup()
    time.sleep(UI_DELAY)  # wait for UI to start
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
