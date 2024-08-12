from collections import namedtuple
import pathlib
import shutil
import sys
import jinja2
import matplotlib.pyplot as plt
import numpy as np
import os
import pywinctl
import time

from cereal import messaging, car, log
from msgq.visionipc import VisionIpcServer, VisionStreamType

from cereal.messaging import SubMaster, PubMaster
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.spinner import Spinner
from openpilot.common.text_window import TextWindow
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.test.helpers import with_processes
from openpilot.selfdrive.test.process_replay.vision_meta import meta_from_camera_state

UI_DELAY = 0.5 # may be slower on CI?

NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength

EventName = car.CarEvent.EventName
EVENTS_BY_NAME = {v: k for k, v in EventName.schema.enumerants.items()}


def click(ui, x, y, *args, **kwargs):
  import pyautogui
  pyautogui.click(ui.left + x, ui.top + y, *args, **kwargs)
  time.sleep(UI_DELAY)  # give enough time for the UI to react


def get_ui(window_title="ui"):
  try:
    return pywinctl.getWindowsWithTitle(window_title)[0]
  except Exception as e:
    print(f"failed to find ui window, assuming that it's in the top left (for Xvfb) {e}")
    return namedtuple("bb", ["left", "top", "width", "height"])(0, 0, 2160, 1080)


def setup_common(window_title="ui"):
  sm = SubMaster(["uiDebug"])
  pm = PubMaster(["deviceState", "pandaStates", "controlsState", 'roadCameraState', 'wideRoadCameraState', 'liveLocationKalman'])

  Params().put("DongleId", "123456789012345")
  dat = messaging.new_message('deviceState')
  dat.deviceState.started = True
  dat.deviceState.networkType = NetworkType.cell4G
  dat.deviceState.networkStrength = NetworkStrength.moderate
  dat.deviceState.freeSpacePercent = 80
  dat.deviceState.memoryUsagePercent = 2
  dat.deviceState.cpuTempC = [2,]*3
  dat.deviceState.gpuTempC = [2,]*3
  dat.deviceState.cpuUsagePercent = [2,]*8

  pm.send("deviceState", dat)

  while not sm.valid["uiDebug"]:
    sm.update(1)
  time.sleep(UI_DELAY)  # wait a bit more for the UI to start rendering
  ui = get_ui(window_title)
  return ui, pm


def setup_homescreen():
  return setup_common()

def setup_settings_device():
  ui, pm = setup_common()

  click(ui, 100, 100)
  return ui, pm

def setup_settings_network():
  ui, pm = setup_common()

  setup_settings_device()
  click(ui, 300, 600)
  return ui, pm

def setup_onroad():
  ui, pm = setup_common()

  dat = messaging.new_message('pandaStates', 1)
  dat.pandaStates[0].ignitionLine = True
  dat.pandaStates[0].pandaType = log.PandaState.PandaType.uno

  pm.send("pandaStates", dat)

  d = DEVICE_CAMERAS[("tici", "ar0231")]
  server = VisionIpcServer("camerad")
  server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 40, False, d.fcam.width, d.fcam.height)
  server.create_buffers(VisionStreamType.VISION_STREAM_DRIVER, 40, False, d.dcam.width, d.dcam.height)
  server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, 40, False, d.fcam.width, d.fcam.height)
  server.start_listener()

  time.sleep(0.5) # give time for vipc server to start

  IMG = np.zeros((int(d.fcam.width*1.5), d.fcam.height), dtype=np.uint8)
  IMG_BYTES = IMG.flatten().tobytes()

  cams = ('roadCameraState', 'wideRoadCameraState')

  frame_id = 0
  for cam in cams:
    msg = messaging.new_message(cam)
    cs = getattr(msg, cam)
    cs.frameId = frame_id
    cs.timestampSof = int((frame_id * DT_MDL) * 1e9)
    cs.timestampEof = int((frame_id * DT_MDL) * 1e9)
    cam_meta = meta_from_camera_state(cam)

    pm.send(msg.which(), msg)
    server.send(cam_meta.stream, IMG_BYTES, cs.frameId, cs.timestampSof, cs.timestampEof)
  time.sleep(2)
  return ui, pm

def setup_onroad_sidebar():
  ui, pm = setup_onroad()
  click(ui, 500, 500)
  return ui, pm

def setup_spinner():
  setup_common()
  s = Spinner()
  s.update_progress(30, 100)
  time.sleep(UI_DELAY)
  ui = get_ui("_spinner")
  return ui, None, s

def close_spinner(s):
  s.close()

def setup_text():
  setup_common()
  t = TextWindow("This is a text window.")
  time.sleep(UI_DELAY)
  ui = get_ui("_text")
  return ui, None, t

def close_text(t):
  t.close()

CASES = {
  # "homescreen": (setup_homescreen, None),
  # "settings_device": (setup_settings_device, None),
  # "settings_network": (setup_settings_network, None),
  "onroad": (setup_onroad, None),
  "onroad_sidebar": (setup_onroad_sidebar, None),
  "spinner": (setup_spinner, close_spinner),
  "text": (setup_text, close_text),
}

TEST_DIR = pathlib.Path(__file__).parent

TEST_OUTPUT_DIR = TEST_DIR / "report_1"
SCREENSHOTS_DIR = TEST_OUTPUT_DIR / "screenshots"


class TestUI:
  def __init__(self):
    os.environ["SCALE"] = "1"
    sys.modules["mouseinfo"] = False

  # def setup(self, window_title):
  #   self.sm = SubMaster(["uiDebug"])
  #   self.pm = PubMaster(["deviceState", "pandaStates", "controlsState", 'roadCameraState', 'wideRoadCameraState', 'liveLocationKalman'])
  #   while not self.sm.valid["uiDebug"]:
  #     self.sm.update(1)
  #   time.sleep(UI_DELAY) # wait a bit more for the UI to start rendering
  #   try:
  #     self.ui = pywinctl.getWindowsWithTitle(window_title)[0]
  #   except Exception as e:
  #     print(f"failed to find ui window, assuming that it's in the top left (for Xvfb) {e}")
  #     self.ui = namedtuple("bb", ["left", "top", "width", "height"])(0,0,2160,1080)

  def screenshot(self, ui):
    import pyautogui
    im = pyautogui.screenshot(region=(ui.left, ui.top, ui.width, ui.height))
    assert im.width == 2160
    assert im.height == 1080
    img = np.array(im)
    im.close()
    return img

  # def click(self, ui, x, y, *args, **kwargs):
  #   import pyautogui
  #   pyautogui.click(ui.left + x, ui.top + y, *args, **kwargs)
  #   time.sleep(UI_DELAY) # give enough time for the UI to react

  @with_processes(["ui"])
  def test_ui(self, name, setup_case, cleanup_case):
    ui, pm, *args = setup_case()

    time.sleep(UI_DELAY) # wait a bit more for the UI to finish rendering

    im = self.screenshot(ui)
    plt.imsave(SCREENSHOTS_DIR / f"{name}.png", im)

    if cleanup_case is not None:
      cleanup_case(*args)


def create_html_report():
  OUTPUT_FILE = TEST_OUTPUT_DIR / "index.html"

  with open(TEST_DIR / "template.html") as f:
    template = jinja2.Template(f.read())

  cases = {f.stem: (str(f.relative_to(TEST_OUTPUT_DIR)), "reference.png") for f in SCREENSHOTS_DIR.glob("*.png")}
  cases = dict(sorted(cases.items()))

  with open(OUTPUT_FILE, "w") as f:
    f.write(template.render(cases=cases))

def create_screenshots():
  if TEST_OUTPUT_DIR.exists():
    shutil.rmtree(TEST_OUTPUT_DIR)

  SCREENSHOTS_DIR.mkdir(parents=True)

  t = TestUI()
  for name, (setup, cleanup) in CASES.items():
    t.test_ui(name, setup, cleanup)

if __name__ == "__main__":
  print("creating test screenshots")
  create_screenshots()

  print("creating html report")
  create_html_report()
