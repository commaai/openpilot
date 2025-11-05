#!/usr/bin/env python3
import os
import sys
import shutil
import time
import pathlib
import subprocess

import pyautogui
import pywinctl

from collections import namedtuple
from collections.abc import Callable

from cereal import car, log
from cereal import messaging
from cereal.messaging import PubMaster
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.test.helpers import with_processes
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from openpilot.system.updated.updated import parse_release_notes

AlertSize = log.SelfdriveState.AlertSize
AlertStatus = log.SelfdriveState.AlertStatus

TEST_DIR = pathlib.Path(__file__).parent
TEST_OUTPUT_DIR = TEST_DIR / "raylib_report"
SCREENSHOTS_DIR = TEST_OUTPUT_DIR / "screenshots"
UI_DELAY = 0.5

BRANCH_NAME = "this-is-a-really-super-mega-ultra-max-extreme-ultimate-long-branch-name"
VERSION = f"0.10.1 / {BRANCH_NAME} / 7864838 / Oct 03"

# Offroad alerts to test
OFFROAD_ALERTS = ['Offroad_IsTakingSnapshot']


def put_update_params(params: Params):
  params.put("UpdaterCurrentReleaseNotes", parse_release_notes(BASEDIR))
  params.put("UpdaterNewReleaseNotes", parse_release_notes(BASEDIR))
  params.put("UpdaterTargetBranch", BRANCH_NAME)


def setup_homescreen(click, pm: PubMaster):
  pass


def setup_homescreen_update_available(click, pm: PubMaster):
  params = Params()
  params.put_bool("UpdateAvailable", True)
  put_update_params(params)
  setup_offroad_alert(click, pm)


def setup_settings(click, pm: PubMaster):
  click(100, 100)


def close_settings(click, pm: PubMaster):
  click(240, 216)


def setup_settings_network(click, pm: PubMaster):
  setup_settings(click, pm)
  click(278, 450)


def setup_settings_network_advanced(click, pm: PubMaster):
  setup_settings_network(click, pm)
  click(1880, 100)


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


def setup_settings_software_branch_switcher(click, pm: PubMaster):
  setup_settings_software(click, pm)
  params = Params()
  params.put("UpdaterAvailableBranches", f"master,nightly,release,{BRANCH_NAME}")
  params.put("GitBranch", BRANCH_NAME)  # should be on top
  params.put("UpdaterTargetBranch", "nightly")  # should be selected
  click(1984, 449)


def setup_settings_firehose(click, pm: PubMaster):
  setup_settings(click, pm)
  click(278, 845)


def setup_settings_developer(click, pm: PubMaster):
  CP = car.CarParams()
  CP.alphaLongitudinalAvailable = True  # show alpha long control toggle
  Params().put("CarParamsPersistent", CP.to_bytes())

  setup_settings(click, pm)
  click(278, 950)


def setup_keyboard(click, pm: PubMaster):
  setup_settings_developer(click, pm)
  click(1930, 470)


def setup_pair_device(click, pm: PubMaster):
  click(1950, 800)


def setup_offroad_alert(click, pm: PubMaster):
  put_update_params(Params())
  set_offroad_alert("Offroad_TemperatureTooHigh", True, extra_text='99C')
  set_offroad_alert("Offroad_ExcessiveActuation", True, extra_text='longitudinal')
  for alert in OFFROAD_ALERTS:
    set_offroad_alert(alert, True)

  setup_settings(click, pm)
  close_settings(click, pm)


def setup_confirmation_dialog(click, pm: PubMaster):
  setup_settings(click, pm)
  click(1985, 791)  # reset calibration


def setup_experimental_mode_description(click, pm: PubMaster):
  setup_settings_toggles(click, pm)
  click(1200, 280)  # expand description for experimental mode


def setup_openpilot_long_confirmation_dialog(click, pm: PubMaster):
  setup_settings_developer(click, pm)
  click(2000, 960)  # toggle openpilot longitudinal control


def setup_onroad(click, pm: PubMaster):
  ds = messaging.new_message('deviceState')
  ds.deviceState.started = True

  ps = messaging.new_message('pandaStates', 1)
  ps.pandaStates[0].pandaType = log.PandaState.PandaType.dos
  ps.pandaStates[0].ignitionLine = True

  driverState = messaging.new_message('driverStateV2')
  driverState.driverStateV2.leftDriverData.faceOrientation = [0, 0, 0]

  for _ in range(5):
    pm.send('deviceState', ds)
    pm.send('pandaStates', ps)
    pm.send('driverStateV2', driverState)
    ds.clear_write_flag()
    ps.clear_write_flag()
    driverState.clear_write_flag()
    time.sleep(0.05)


def setup_onroad_sidebar(click, pm: PubMaster):
  setup_onroad(click, pm)
  click(100, 100)  # open sidebar


def setup_onroad_alert(click, pm: PubMaster, size: log.SelfdriveState.AlertSize, text1: str, text2: str, status: log.SelfdriveState.AlertStatus):
  setup_onroad(click, pm)
  alert = messaging.new_message('selfdriveState')
  ss = alert.selfdriveState
  ss.alertSize = size
  ss.alertText1 = text1
  ss.alertText2 = text2
  ss.alertStatus = status
  for _ in range(5):
    pm.send('selfdriveState', alert)
    alert.clear_write_flag()
    time.sleep(0.05)


def setup_onroad_small_alert(click, pm: PubMaster):
  setup_onroad_alert(click, pm, AlertSize.small, "Small Alert", "This is a small alert", AlertStatus.normal)


def setup_onroad_medium_alert(click, pm: PubMaster):
  setup_onroad_alert(click, pm, AlertSize.mid, "Medium Alert", "This is a medium alert", AlertStatus.userPrompt)


def setup_onroad_full_alert(click, pm: PubMaster):
  setup_onroad_alert(click, pm, AlertSize.full, "DISENGAGE IMMEDIATELY", "Driver Distracted", AlertStatus.critical)


def setup_onroad_full_alert_multiline(click, pm: PubMaster):
  setup_onroad_alert(click, pm, AlertSize.full, "Reverse\nGear", "", AlertStatus.normal)


def setup_onroad_full_alert_long_text(click, pm: PubMaster):
  setup_onroad_alert(click, pm, AlertSize.full, "TAKE CONTROL IMMEDIATELY", "Calibration Invalid: Remount Device & Recalibrate", AlertStatus.userPrompt)


CASES: dict[str, Callable] = {
  "homescreen": setup_homescreen,
  "homescreen_paired": setup_homescreen,
  "homescreen_prime": setup_homescreen,
  "homescreen_update_available": setup_homescreen_update_available,
  "homescreen_unifont": setup_homescreen,
  "settings_device": setup_settings,
  "settings_network": setup_settings_network,
  "settings_network_advanced": setup_settings_network_advanced,
  "settings_toggles": setup_settings_toggles,
  "settings_software": setup_settings_software,
  "settings_software_download": setup_settings_software_download,
  "settings_software_release_notes": setup_settings_software_release_notes,
  "settings_software_branch_switcher": setup_settings_software_branch_switcher,
  "settings_firehose": setup_settings_firehose,
  "settings_developer": setup_settings_developer,
  "keyboard": setup_keyboard,
  "pair_device": setup_pair_device,
  "offroad_alert": setup_offroad_alert,
  "confirmation_dialog": setup_confirmation_dialog,
  "experimental_mode_description": setup_experimental_mode_description,
  "openpilot_long_confirmation_dialog": setup_openpilot_long_confirmation_dialog,
  "onroad": setup_onroad,
  "onroad_sidebar": setup_onroad_sidebar,
  "onroad_small_alert": setup_onroad_small_alert,
  "onroad_medium_alert": setup_onroad_medium_alert,
  "onroad_full_alert": setup_onroad_full_alert,
  "onroad_full_alert_multiline": setup_onroad_full_alert_multiline,
  "onroad_full_alert_long_text": setup_onroad_full_alert_long_text,
}


def fullscreen_click_primary_button(click, pm: PubMaster):
  click(1950, 950)  # Bottom right button


def fullscreen_click_secondary_button(click, pm: PubMaster):
  click(150, 950)  # Bottom left button


def software_setup_get_started_next(click, pm: PubMaster):
  click(2000, 630)


def software_setup_choose_software_click_openpilot(click, pm: PubMaster):
  click(1200, 320)


def software_setup_choose_software_click_custom(click, pm: PubMaster):
  click(1200, 580)


# These cases are for the setup, updater, and reset screens that have their own UI process.
# The key is the name of the script.
# Each case is a list of additional steps to perform and screenshot (after initial screenshot).
# Each item can also be a group of steps to do, with the screenshot at the end.
SOFTWARE_SETUP_CASES: dict[str, list | list[list]] = {
  "setup": [
    fullscreen_click_primary_button,  # Low voltage warning; click "Continue"
    software_setup_get_started_next,  # Get started page; click arrow
    [
      # Take a screenshot of the custom software warning first, so we can go back
      software_setup_choose_software_click_custom,  # Click "Custom" on choose software page
      fullscreen_click_primary_button,  # Click "Continue"
    ],
    [fullscreen_click_secondary_button, software_setup_choose_software_click_openpilot],  # Go back to choose software page and click "openpilot"
    [fullscreen_click_primary_button, lambda click, pm: time.sleep(1)],  # Click "Continue"; wait for networks to load
    fullscreen_click_primary_button,  # "Download" button
  ],
  "updater": [
    fullscreen_click_secondary_button,  # Click "Connect to Wi-Fi"
    [fullscreen_click_secondary_button, fullscreen_click_primary_button],  # Click "Back", then "Install"
  ],
  "reset": [
    fullscreen_click_primary_button,  # Click "Confirm" on initial confirmation
    fullscreen_click_primary_button,  # Click "Confirm" on final warning
  ],
}


class TestUI:
  def __init__(self, window_title="UI"):
    self.window_title = window_title

    os.environ["SCALE"] = os.getenv("SCALE", "1")
    sys.modules["mouseinfo"] = False

  def setup(self):
    # Seed minimal offroad state
    self.pm = PubMaster(["deviceState", "pandaStates", "driverStateV2", "selfdriveState"])
    ds = messaging.new_message('deviceState')
    ds.deviceState.networkType = log.DeviceState.NetworkType.wifi
    for _ in range(5):
      self.pm.send('deviceState', ds)
      ds.clear_write_flag()
      time.sleep(0.05)
    time.sleep(0.5)
    try:
      self.ui = pywinctl.getWindowsWithTitle(self.window_title)[0]
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
  def test_ui(self, name, setup_case: Callable):
    self.setup()
    time.sleep(UI_DELAY)  # wait for UI to start
    setup_case(self.click, self.pm)
    self.screenshot(name)


class TestScriptUI(TestUI):
  def __init__(self, script_path: str, script_args: list[str] | None, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._script_path = script_path
    self._script_args = script_args or []
    self._process = None

  def __enter__(self):
    self._process = subprocess.Popen([sys.executable, self._script_path] + self._script_args)
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    if self._process:
      self._process.terminate()
      try:
        self._process.wait(timeout=5)
      except subprocess.TimeoutExpired:
        self._process.kill()
      self._process = None

  # Override the TestUI method to run multiple tests, and to avoid starting another UI process
  def test_ui(self, name, setup_cases: list[Callable | list[Callable]]):
    self.setup()
    time.sleep(UI_DELAY)
    self.screenshot(name)  # initial screenshot
    # Run each setup case, taking a screenshot after each group
    for i, case in enumerate(setup_cases):
      group = case if isinstance(case, list) else [case]  # each case can be a single step or group of steps
      for setup_case in group:
        setup_case(self.click, self.pm)  # run each step in the group
        time.sleep(0.1)  # allow UI to update between steps
      self.screenshot(f"{name}_{i + 2}")  # take screenshot after each case group


def create_screenshots():
  if TEST_OUTPUT_DIR.exists():
    shutil.rmtree(TEST_OUTPUT_DIR)
  SCREENSHOTS_DIR.mkdir(parents=True)

  t = TestUI()
  for name, setup in CASES.items():
    with OpenpilotPrefix():
      params = Params()
      params.put("DongleId", "123456789012345")

      # Set branch name
      params.put("UpdaterCurrentDescription", VERSION)
      params.put("UpdaterNewDescription", VERSION)

      if name == "homescreen_paired":
        params.put("PrimeType", 0)  # NONE
      elif name == "homescreen_prime":
        params.put("PrimeType", 2)  # LITE
      elif name == "homescreen_unifont":
        params.put("LanguageSetting", "zh-CHT")  # Traditional Chinese

      t.test_ui(name, setup)

  for name, setup_cases in SOFTWARE_SETUP_CASES.items():
    with OpenpilotPrefix():
      window_title = "System Reset" if name == "reset" else name.capitalize()
      args = ["updater", "manifest"] if name == "updater" else None
      with TestScriptUI(f"system/ui/{name}.py", args, window_title=window_title) as launcher:
        launcher.test_ui(name, setup_cases)


if __name__ == "__main__":
  create_screenshots()
