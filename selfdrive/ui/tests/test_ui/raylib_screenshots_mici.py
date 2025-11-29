#!/usr/bin/env python3
from cereal.messaging import PubMaster
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.system.version import terms_version, training_version
from openpilot.system.ui.lib.application import gui_app
from selfdrive.ui.tests.test_ui.raylib_screenshots import TestUI, SCREENSHOTS_DIR, TEST_OUTPUT_DIR, VERSION


def setup_mici_homescreen(click, pm: PubMaster):
  pass


def setup_mici_settings(click, pm: PubMaster):
  click(1, 1)


def setup_mici_settings_toggles(click, pm: PubMaster):
  setup_mici_settings(click, pm)
  click(gui_app.width // 2, gui_app.height // 2)


CASES = {
  "homescreen": setup_mici_homescreen,
  "settings": setup_mici_settings,
  "settings_toggles": setup_mici_settings_toggles,
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
