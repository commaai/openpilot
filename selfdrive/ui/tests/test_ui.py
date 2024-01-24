import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pyautogui
import pywinctl
import time
import unittest

from parameterized import parameterized

from cereal.messaging import SubMaster
from openpilot.selfdrive.test.helpers import with_processes

UI_DELAY = 0.5 # may be slower on CI?

def setup_homescreen(click):
  pass

def setup_settings_device(click):
  click(100, 100)
  time.sleep(UI_DELAY)

def setup_settings_network(click):
  setup_settings_device(click)
  click(300, 600)
  time.sleep(UI_DELAY)


TEST_OUTPUT_DIR = pathlib.Path(__file__).parent / "test_ui_screenshots"


class TestUI(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    os.environ["SCALE"] = "1"

  def setup(self):
    self.sm = SubMaster(["uiDebug"])
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

  @parameterized.expand([
    ("homescreen", setup_homescreen),
    ("settings", setup_settings_device),
    ("network", setup_settings_network),
  ])
  @with_processes(["ui"])
  def test_ui(self, name, setup_case):
    self.setup()

    setup_case(self.click)

    im = self.screenshot()
    plt.imsave(TEST_OUTPUT_DIR / f"{name}.png", im)


if __name__ == "__main__":
  unittest.main()
