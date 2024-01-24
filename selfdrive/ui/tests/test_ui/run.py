import pathlib
import shutil
import jinja2
import matplotlib.pyplot as plt
import numpy as np
import os
import pyautogui
import pywinctl
import time
import unittest

from parameterized import parameterized
from cereal import messaging, log

from cereal.messaging import SubMaster, PubMaster
from openpilot.common.params import Params
from openpilot.selfdrive.test.helpers import with_processes

UI_DELAY = 0.5 # may be slower on CI?

NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength

def setup_common(click, pm: PubMaster):
  Params().put("DongleId", "123456789012345")
  dat = messaging.new_message('deviceState')
  dat.deviceState.networkType = NetworkType.cell4G
  dat.deviceState.networkStrength = NetworkStrength.moderate

  pm.send("deviceState", dat)

  time.sleep(UI_DELAY)

def setup_homescreen(click, pm: PubMaster):
  setup_common(click, pm)

def setup_settings_device(click, pm: PubMaster):
  setup_common(click, pm)

  click(100, 100)
  time.sleep(UI_DELAY)

def setup_settings_network(click, pm: PubMaster):
  setup_common(click, pm)

  setup_settings_device(click, pm)
  click(300, 600)
  time.sleep(UI_DELAY)

CASES = {
  "homescreen": setup_homescreen,
  "settings_device": setup_settings_device,
  "settings_network": setup_settings_network,
}


TEST_DIR = pathlib.Path(__file__).parent

TEST_OUTPUT_DIR = TEST_DIR / "report"
SCREENSHOTS_DIR = TEST_OUTPUT_DIR / "screenshots"


class TestUI(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    os.environ["SCALE"] = "1"

  def setup(self):
    self.sm = SubMaster(["uiDebug"])
    self.pm = PubMaster(["deviceState"])
    while not self.sm.valid["uiDebug"]:
      self.sm.update(1)
    time.sleep(UI_DELAY) # wait a bit more for the UI to finish rendering
    self.ui = pywinctl.getWindowsWithTitle("ui")[0]

  def screenshot(self):
    im = pyautogui.screenshot(region=(self.ui.left, self.ui.top, self.ui.width, self.ui.height))
    self.assertEqual(im.width, 2160)
    self.assertEqual(im.height, 1080)
    img = np.array(im)
    im.close()
    return img

  def click(self, x, y, *args, **kwargs):
    pyautogui.click(self.ui.left + x, self.ui.top + y, *args, **kwargs)

  @parameterized.expand(CASES.items())
  @with_processes(["ui"])
  def test_ui(self, name, setup_case):
    self.setup()

    setup_case(self.click, self.pm)

    im = self.screenshot()
    plt.imsave(SCREENSHOTS_DIR / f"{name}.png", im)


def create_html_report():
  OUTPUT_FILE = TEST_OUTPUT_DIR / "index.html"

  with open(TEST_DIR / "template.html") as f:
    template = jinja2.Template(f.read())

  cases = {f.stem: (str(f.relative_to(TEST_OUTPUT_DIR)), "reference.png") for f in SCREENSHOTS_DIR.glob("*.png")}

  with open(OUTPUT_FILE, "w") as f:
    f.write(template.render(cases=cases))

def create_screenshots():
  if TEST_OUTPUT_DIR.exists():
    shutil.rmtree(TEST_OUTPUT_DIR)

  SCREENSHOTS_DIR.mkdir(parents=True)
  unittest.main(exit=False)

if __name__ == "__main__":
  print("creating test screenshots")
  create_screenshots()

  print("creating html report")
  create_html_report()
