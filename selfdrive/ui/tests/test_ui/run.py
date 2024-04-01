from collections import namedtuple
import pathlib
import shutil
import sys
import jinja2
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pywinctl
import time
import unittest

from parameterized import parameterized
from cereal import messaging, car, log
from cereal.visionipc import VisionIpcServer, VisionStreamType

from cereal.messaging import SubMaster, PubMaster
from openpilot.common.mock import mock_messages
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.test.helpers import with_processes
from openpilot.selfdrive.test.process_replay.vision_meta import meta_from_camera_state
from openpilot.tools.webcam.camera import Camera
from openpilot.system.version import terms_version, training_version

from openpilot.selfdrive.controls.lib.events import EVENTS, Alert
from openpilot.selfdrive.controls.lib.alertmanager import set_offroad_alert, OFFROAD_ALERTS

UI_DELAY = 2 # may be slower on CI?

NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength

State = log.ControlsState.OpenpilotState

EventName = car.CarEvent.EventName
EVENTS_BY_NAME = {v: k for k, v in EventName.schema.enumerants.items()}

available_langs = json.load(open(os.path.normpath(__file__ + "/../../../translations/languages.json"), "r"))
test_langs = ["main_" + lang for lang in (os.environ.get("TEST_UI_LANGUAGES") or "").split(" ")]
langs = dict(filter(lambda x: x[1] == "main_en" or x[1] in test_langs, available_langs.items()))

class PrimeType:
  UNKNOWN = b'-1'
  NONE = b'0'
  MAGENTA = b'1'
  LITE = b'2'
  BLUE = b'3'
  MAGENTA_NEW = b'4'
  PURPLE = b'5'

def event_from_id(event_id: int):
  try: # O(1)
    return car.CarEvent.EventName.schema.node.enum.enumerants[event_id].name
  except IndexError:
    return "Unknown Event ID"

def get_onroad_alert_cases():
  cases = {}
  for event_id, event in EVENTS.items():
    event_name = event_from_id(event_id)
    for event_type, alert in event.items():
      if callable(alert):
        continue

      case_name = f"onroad_alert_{event_name}_{event_type}"
      alert.event_type = event_type
      alert.alert_type = event_name
      cases[case_name] = (lambda click, pm, alert=alert:
                          send_onroad_alert(click, pm, alert))
  return cases

def send_onroad_alert(click, pm, alert: Alert):
  dat = messaging.new_message('controlsState', valid=True)
  dat.controlsState.enabled = True
  dat.controlsState.state = State.enabled if alert.event_type is None or 'override' not in alert.event_type else State.overriding
  dat.controlsState.alertText1 = alert.alert_text_1
  dat.controlsState.alertText2 = alert.alert_text_2
  dat.controlsState.alertSize = alert.alert_size
  dat.controlsState.alertStatus = alert.alert_status
  dat.controlsState.alertBlinkingRate = alert.alert_rate
  dat.controlsState.alertType = alert.alert_type
  dat.controlsState.alertSound = alert.audible_alert
  pm.wait_for_readers_to_update('controlsState', UI_DELAY)
  pm.send('controlsState', dat)

def setup_common(click, pm: PubMaster):
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

def setup_keyboard(click, pm):
  setup_settings_network_advanced(click, pm)
  click(1900, 350) # Edit tethering password

def setup_numbers_keyboard(click, pm):
  setup_keyboard(click, pm)
  click(150, 950)

def setup_homescreen(click, pm: PubMaster):
  setup_common(click, pm)

def setup_settings_device(click, pm: PubMaster):
  setup_common(click, pm)

  click(100, 100)

def setup_settings_network(click, pm: PubMaster):
  setup_common(click, pm)

  setup_settings_device(click, pm)
  click(300, 600)

def setup_settings_network_advanced(click, pm: PubMaster):
  setup_settings_network(click, pm)
  click(2000, 50)

def setup_settings_software(click, pm: PubMaster):
  setup_common(click, pm)
  setup_settings_device(click, pm)
  click(300, 950)

def setup_settings_toggles(click, pm: PubMaster):
  setup_common(click, pm)
  setup_settings_device(click, pm)
  click(300, 750)

def setup_offroad_driver_camera(click, pm: PubMaster):
  setup_common(click, pm)
  setup_settings_device(click, pm)
  click(1900,450)

def setup_experimental_mode_prompt(click, pm: PubMaster):
  setup_settings_toggles(click, pm)
  click(2000, 250)

def setup_update(click, pm: PubMaster):
  setup_common(click, pm)
  Params().put_bool("UpdateAvailable", True)
  Params().put("UpdaterNewDescription", "A quick brown fox jumps over the lazy dog .")
  Params().put("UpdaterNewReleaseNotes", "A quick brown fox jumps over the lazy dog .")
  click(100, 100) # Open and close settings to refresh
  click(250, 250)
  Params().put_bool("UpdateAvailable", False)

def setup_onroad(click, pm: PubMaster):
  setup_common(click, pm)

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

  IMG = Camera.bgr2nv12(np.random.randint(0, 255, (d.fcam.width, d.fcam.height, 3), dtype=np.uint8))
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

def setup_controls_enabled(click, pm: PubMaster, state = State.enabled):
  msg = messaging.new_message('controlsState', valid=True)
  msg.controlsState.enabled = True
  msg.controlsState.state = state
  pm.send('controlsState', msg)

def setup_navoop(click, pm: PubMaster):
  msg = messaging.new_message('navInstruction', valid=True)
  pm.send('navInstruction', msg)
  msg = messaging.new_message('navRoute', valid=True)
  pm.send('navRoute', msg)
  modelv2_send = messaging.new_message('modelV2')
  modelv2_send.modelV2.navEnabled = True
  pm.wait_for_readers_to_update('modelV2',UI_DELAY)
  pm.send('modelV2', modelv2_send)

def setup_onroad_engaged(click, pm: PubMaster):
  setup_onroad(click, pm)
  setup_controls_enabled(click, pm)

def setup_onroad_overriding(click, pm: PubMaster):
  setup_onroad(click, pm)
  setup_controls_enabled(click, pm, State.overriding)

def setup_onroad_nav_enabled(click, pm: PubMaster):
  setup_onroad_sidebar(click, pm)
  setup_controls_enabled(click, pm)
  setup_navoop(click, pm)
  time.sleep(UI_DELAY) # give time for the map to render

@mock_messages(['liveLocationKalman'])
def setup_onroad_map(click, pm: PubMaster):
  setup_onroad(click, pm)

  click(500, 500)

  time.sleep(UI_DELAY) # give time for the map to render

def setup_onroad_sidebar(click, pm: PubMaster):
  setup_onroad_map(click, pm)
  click(500, 500)

def send_offroad_alert(click, offroad_alert):
  set_offroad_alert(offroad_alert[0], False)
  set_offroad_alert(offroad_alert[0], True)
  click(100, 100) # Open and close settings to refresh
  click(250, 250)

CASES = {
  "homescreen": (setup_homescreen, {'prime': PrimeType.NONE}),
  "homescreen_with_prime": (setup_homescreen, {'prime': PrimeType.PURPLE}),
  "settings_device": setup_settings_device,
  "settings_network": setup_settings_network,
  "settings_network_advanced": setup_settings_network_advanced,
  "settings_software": setup_settings_software,
  "settings_toggles": setup_settings_toggles,
  "offroad_driver_camera": setup_offroad_driver_camera,
  "update": setup_update,
  "keyboard": setup_keyboard,
  "numbers_keyboard": setup_numbers_keyboard,
  "onroad": setup_onroad,
  "onroad_engaged": setup_onroad_engaged,
  "onroad_overriding": setup_onroad_overriding,
  "onroad_nav_enabled": setup_onroad_nav_enabled,
  "onroad_map": setup_onroad_map,
  "onroad_sidebar": setup_onroad_sidebar,
  "experimental_mode_confirm": setup_experimental_mode_prompt
}

TEST_DIR = pathlib.Path(__file__).parent

TEST_OUTPUT_DIR = TEST_DIR / "report"
SCREENSHOTS_DIR = TEST_OUTPUT_DIR / "screenshots"

class TestUI(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    os.environ["SCALE"] = "1"
    sys.modules["mouseinfo"] = False

  @classmethod
  def tearDownClass(cls):
    del sys.modules["mouseinfo"]

  def setup(self, prime_type = PrimeType.NONE):
    Params().put("HasAcceptedTerms", terms_version)
    Params().put("CompletedTrainingVersion", training_version)
    Params().put("PrimeType", prime_type)
    self.sm = SubMaster(["uiDebug"])
    self.pm = PubMaster(["deviceState", "pandaStates", "controlsState", 'roadCameraState', 'wideRoadCameraState', 'liveLocationKalman',
    'navRoute', 'navInstruction', 'modelV2'])
    while not self.sm.valid["uiDebug"]:
      self.sm.update(1)
    time.sleep(UI_DELAY) # wait a bit more for the UI to start rendering
    try:
      self.ui = pywinctl.getWindowsWithTitle("ui")[0]
    except Exception as e:
      print(f"failed to find ui window, assuming that it's in the top left (for Xvfb) {e}")
      self.ui = namedtuple("bb", ["left", "top", "width", "height"])(0,0,2160,1080)

  def screenshot(self):
    import pyautogui
    im = pyautogui.screenshot(region=(self.ui.left, self.ui.top, self.ui.width, self.ui.height))
    self.assertEqual(im.width, 2160)
    self.assertEqual(im.height, 1080)
    img = np.array(im)
    im.close()
    return img

  def click(self, x, y, *args, **kwargs):
    import pyautogui
    pyautogui.click(self.ui.left + x, self.ui.top + y, *args, **kwargs)
    time.sleep(UI_DELAY) # give enough time for the UI to react

  @parameterized.expand([
    (case_name, setup_data, lang_code, lang_name)
    for case_name, setup_data in CASES.items()
    for lang_name, lang_code in langs.items()
  ])
  def test_ui(self, name, setup_data, lang_code, lang_name, *args):
    Params().put("LanguageSetting", lang_code)
    self.run_ui_test_case(name, setup_data, lang_code, *args)

  @with_processes(["ui"])
  def run_ui_test_case(self, name, setup_data, lang_code, *args):
    setup_info = {}

    if isinstance(setup_data, tuple):
      setup_case, setup_info = setup_data
    else:
      setup_case = setup_data

    self.setup(dict.get(setup_info, "prime", PrimeType.NONE))

    setup_case(self.click, self.pm)

    time.sleep(UI_DELAY) # wait a bit more for the UI to finish rendering

    im = self.screenshot()
    (SCREENSHOTS_DIR / lang_code).mkdir(parents=True, exist_ok=True)
    plt.imsave(SCREENSHOTS_DIR / f"{lang_code}/{name}.png", im)

  @parameterized.expand(list(langs.items()))
  def test_offroad_alerts(self, _, lang_code):
    Params().put("LanguageSetting", lang_code)
    self.run_offroad_alerts(lang_code)

  @with_processes(["ui"])
  def run_offroad_alerts(self, lang_code):
    self.setup()
    setup_common(self.click, self.pm)
    time.sleep(UI_DELAY)
    for offroad_alert in OFFROAD_ALERTS.items():
      send_offroad_alert(self.click, offroad_alert)
      im = self.screenshot()
      set_offroad_alert(offroad_alert[0], False)
      (SCREENSHOTS_DIR / lang_code / "alerts").mkdir(parents=True, exist_ok=True)
      plt.imsave(SCREENSHOTS_DIR / f"{lang_code}/alerts/{offroad_alert[0]}.png", im)

  @parameterized.expand(list(langs.items()))
  def test_ui_events(self, _, lang_code):
    Params().put("LanguageSetting", lang_code)
    self.run_events(lang_code)

  @with_processes(["ui"])
  def run_events(self, lang_code):
    cases = get_onroad_alert_cases()
    self.setup()
    setup_onroad(self.click, self.pm)
    time.sleep(UI_DELAY)
    for name, setup_case in cases.items():
      setup_case(self.click, self.pm)
      time.sleep(UI_DELAY)
      im = self.screenshot()
      (SCREENSHOTS_DIR / lang_code / "alerts").mkdir(parents=True, exist_ok=True)
      plt.imsave(SCREENSHOTS_DIR / f"{lang_code}/alerts/{name}.png", im)


def create_html_report():
  OUTPUT_FILE = TEST_OUTPUT_DIR / "index.html"

  with open(TEST_DIR / "template.html") as f:
    template = jinja2.Template(f.read())

  cases = {("alerts/" if f.parent.name == "alerts" else "") + f.stem: (str(f.relative_to(TEST_OUTPUT_DIR)), "reference.png") for f in SCREENSHOTS_DIR.glob("main_en/**/*.png")}
  cases = dict(sorted(cases.items()))

  with open(OUTPUT_FILE, "w") as f:
    f.write(template.render(cases=cases, langs=langs))

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
