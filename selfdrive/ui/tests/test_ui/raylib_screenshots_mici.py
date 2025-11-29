#!/usr/bin/env python3
"""
MICI UI screenshot tests.

This file imports common utilities from raylib_screenshots.py and defines
MICI-specific test cases for the 536x240 resolution UI.
"""
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.system.version import terms_version, training_version

# Import common utilities and classes
from selfdrive.ui.tests.test_ui.raylib_screenshots import (
  TestUI, SCREENSHOTS_DIR, TEST_OUTPUT_DIR, UI_DELAY,
  BRANCH_NAME, VERSION, OFFROAD_ALERTS,
  put_update_params, setup_onroad_state, setup_onroad_alert_state,
  AlertSize, AlertStatus, set_offroad_alert,
)
from cereal.messaging import PubMaster


# MICI UI setup functions (536x240 resolution)
def setup_mici_homescreen(click, pm: PubMaster):
  pass


def setup_mici_homescreen_update_available(click, pm: PubMaster):
  params = Params()
  params.put_bool("UpdateAvailable", True)
  put_update_params(params)
  setup_mici_offroad_alert(click, pm)


def setup_mici_settings(click, pm: PubMaster):
  # Click settings button (top right, ~48x48 icon)
  click(488, 8)


def setup_mici_settings_toggles(click, pm: PubMaster):
  setup_mici_settings(click, pm)
  # Click toggles button (first button in scroller)
  click(268, 120)


def setup_mici_settings_network(click, pm: PubMaster):
  setup_mici_settings(click, pm)
  # Click network button (second button in scroller)
  click(268, 180)


def setup_mici_settings_device(click, pm: PubMaster):
  setup_mici_settings(click, pm)
  # Click device button (third button in scroller)
  click(268, 120)


def setup_mici_settings_developer(click, pm: PubMaster):
  setup_mici_settings(click, pm)
  # Scroll to developer button (last button in scroller)
  click(268, 200)


def setup_mici_keyboard(click, pm: PubMaster):
  setup_mici_settings_developer(click, pm)
  # Click keyboard button in developer panel
  click(268, 120)


def setup_mici_offroad_alert(click, pm: PubMaster):
  put_update_params(Params())
  set_offroad_alert("Offroad_TemperatureTooHigh", True, extra_text='99C')
  set_offroad_alert("Offroad_ExcessiveActuation", True, extra_text='longitudinal')
  for alert in OFFROAD_ALERTS:
    set_offroad_alert(alert, True)
  setup_mici_settings(click, pm)
  # Close settings to show alerts
  click(8, 8)  # Back button


def setup_mici_onroad(click, pm: PubMaster):
  setup_onroad_state(pm)


def setup_mici_onroad_full_alert(click, pm: PubMaster):
  setup_mici_onroad(click, pm)
  setup_onroad_alert_state(pm, AlertSize.full, "DISENGAGE IMMEDIATELY", "Driver Distracted", AlertStatus.critical)


CASES = {
  "homescreen": setup_mici_homescreen,
  "homescreen_update_available": setup_mici_homescreen_update_available,
  "settings_device": setup_mici_settings_device,
  "settings_network": setup_mici_settings_network,
  "settings_toggles": setup_mici_settings_toggles,
  "settings_developer": setup_mici_settings_developer,
  "keyboard": setup_mici_keyboard,
  "offroad_alert": setup_mici_offroad_alert,
  "onroad": setup_mici_onroad,
  "onroad_full_alert": setup_mici_onroad_full_alert,
}


def create_screenshots():
  if TEST_OUTPUT_DIR.exists():
    import shutil
    shutil.rmtree(TEST_OUTPUT_DIR)
  SCREENSHOTS_DIR.mkdir(parents=True)

  t = TestUI(big_ui=False)
  for name, setup in CASES.items():
    with OpenpilotPrefix():
      params = Params()
      params.put("DongleId", "123456789012345")

      # Set branch name
      params.put("UpdaterCurrentDescription", VERSION)
      params.put("UpdaterNewDescription", VERSION)

      # Set terms and training version (to skip onboarding)
      params.put("HasAcceptedTerms", terms_version)
      params.put("CompletedTrainingVersion", training_version)

      t.test_ui(f"mici_{name}", setup)


if __name__ == "__main__":
  create_screenshots()
