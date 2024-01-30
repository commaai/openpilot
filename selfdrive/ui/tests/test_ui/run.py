from collections import namedtuple
from typing import List, Dict, Callable, Any, Optional
import pathlib
import shutil
import sys
import jinja2
import matplotlib.pyplot as plt
import numpy as np
import os
import pywinctl
import time
import unittest

from parameterized import parameterized
from cereal import messaging, car, log
from cereal.visionipc import VisionIpcServer, VisionStreamType

from cereal.messaging import SubMaster, PubMaster
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.transformations.camera import tici_f_frame_size
from openpilot.selfdrive.navd.tests.test_map_renderer import gen_llk
from openpilot.selfdrive.test.helpers import with_processes
from openpilot.selfdrive.test.process_replay.vision_meta import meta_from_camera_state
from openpilot.selfdrive.controls.lib.events import EVENTS, Alert, ET
from openpilot.selfdrive.controls.lib.alertmanager import set_offroad_alert, OFFROAD_ALERTS

from openpilot.tools.webcam.camera import Camera

UI_DELAY = .5 # may be slower on CI?

NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength
EventName = car.CarEvent.EventName
State = log.ControlsState.OpenpilotState

class PrimeType:
  UNKNOWN = '-1'
  NONE = '0'
  MAGENTA = '1'
  LITE = '2'
  BLUE = '3'
  MAGENTA_NEW = '4'
  PURPLE = '5'

def event_from_id(event_id: int) -> str:
  try: # O(1)
    return car.CarEvent.EventName.schema.node.enum.enumerants[event_id].name
  except IndexError:
    return "Unknown Event ID"

def setup_common(click, pm: PubMaster):
  Params().put("DongleId", "123456789012345")
  Params().put_bool("UpdateAvailable", False)
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

def setup_homescreen(click, pm: PubMaster):
  setup_common(click, pm)

def setup_settings_device(click, pm: PubMaster):
  setup_common(click, pm)

  click(100, 100)

def setup_settings_network(click, pm: PubMaster):
  setup_common(click, pm)

  setup_settings_device(click, pm)
  click(300, 600)

def setup_onroad(click, pm: PubMaster):
  setup_common(click, pm)

  dat = messaging.new_message('pandaStates', 1)
  dat.pandaStates[0].ignitionLine = True
  dat.pandaStates[0].pandaType = log.PandaState.PandaType.uno

  pm.send("pandaStates", dat)

  server = VisionIpcServer("camerad")
  server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 40, False, *tici_f_frame_size)
  server.create_buffers(VisionStreamType.VISION_STREAM_DRIVER, 40, False, *tici_f_frame_size)
  server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, 40, False, *tici_f_frame_size)
  server.start_listener()

  time.sleep(0.5) # give time for vipc server to start

  IMG = Camera.bgr2nv12(np.random.randint(0, 255, (*tici_f_frame_size,3), dtype=np.uint8))
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

def setup_onroad_map(click, pm: PubMaster):
  setup_onroad(click, pm)

  dat = gen_llk()
  pm.send("liveLocationKalman", dat)

  click(500, 500)

  time.sleep(UI_DELAY) # give time for the map to render

def setup_onroad_sidebar(click, pm: PubMaster):
  setup_onroad_map(click, pm)
  click(500, 500)

def setup_controls_enabled(click, pm: PubMaster):
  msg = messaging.new_message('controlsState', valid=True)
  msg.controlsState.enabled = True
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

def setup_onroad_nav_enabled(click, pm: PubMaster):
  setup_onroad_sidebar(click, pm) # Show the sidebar before setting navoop. Frogpilot got this wrong before :P
  setup_controls_enabled(click, pm)
  setup_navoop(click, pm)
  time.sleep(UI_DELAY) # give time for the map to render

def setup_update(click, pm: PubMaster):
  setup_common(click, pm)
  Params().put_bool("UpdateAvailable", True)
  Params().put("UpdaterNewDescription", "UpdaterNewDescription")
  Params().put("UpdaterNewReleaseNotes", "UpdaterNewReleaseNotes")
  click(100, 100) # Open and close settings to refresh
  click(250, 250)
  Params().put_bool("UpdateAvailable", False)

def setup_update_close(click, pm: PubMaster):
  setup_update(click, pm)
  click(550,900)

def get_onroad_alert_cases() -> Dict[str, Callable[[Callable[[int, int], None], PubMaster, Alert], None]]:
  cases = {}
  allowed_event_names = ["ALL"]
  allowed_event_types = ["ALL",ET.OVERRIDE_LATERAL,ET.OVERRIDE_LONGITUDINAL]

  for event_id, event in EVENTS.items():
    event_name = event_from_id(event_id)
    for event_type, alert in event.items():
      if callable(alert):
        continue

      case_name = f"onroad_alert_{event_name}_{event_type}"
      if ("ALL" in allowed_event_types or event_type in allowed_event_types) and \
        ("ALL" in allowed_event_names or event_name in allowed_event_names):
        alert.event_type = event_type
        alert.alert_type = event_name
        cases[case_name] = (lambda click, pm, alert=alert:
                            send_onroad_alert(click, pm, alert))

  return cases

def get_offroad_alert_cases():
  for offroad_alert in OFFROAD_ALERTS.items():
    print(offroad_alert[0])
    print(offroad_alert[1]['text'])
    set_offroad_alert(offroad_alert[0], True,offroad_alert[1]['text'])
    #print(offroad_alert)

def set_prime(prime_type: PrimeType = PrimeType.PURPLE):
  try:
    Params().put("PrimeType", prime_type)
  except Exception as e:
    print(e)

def send_onroad_alert(click, pm, alert: Alert):
  print(alert.event_type, alert)
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
  pm.wait_for_readers_to_update('controlsState',UI_DELAY)
  pm.send('controlsState', dat)

def send_offroad_alert(click, pm, offroad_alert):
  set_offroad_alert(offroad_alert[0], False)
  print(f'Testing offroad alert: {offroad_alert[0]}')
  print(offroad_alert[1]['text'])
  set_offroad_alert(offroad_alert[0], True)
  click(100, 100) # Open and close settings to refresh
  click(250, 250)


BASE_CASES: List[tuple[str,tuple[Callable[[...], None],Optional[PrimeType]]]]= [
  ("homescreen", [setup_homescreen, PrimeType.MAGENTA]),
  ("homescreen", setup_homescreen),
  ("onroad_nav_enabled", (setup_onroad_nav_enabled, PrimeType.MAGENTA)),
  ("onroad_nav_enabled", setup_onroad_nav_enabled),
  ("onroad_map", (setup_onroad_map, PrimeType.MAGENTA)),
  ("onroad_map", setup_onroad_map),
  ("settings_device", setup_settings_device),
  ("settings_network", setup_settings_network),
  ("onroad", setup_onroad),
  ("onroad_sidebar", setup_onroad_sidebar),
  ("offroad_update", setup_update),
  ("offroad_update_confirm", setup_update_close),
]

TEST_DIR = pathlib.Path(__file__).parent

TEST_OUTPUT_DIR = TEST_DIR / "report"
SCREENSHOTS_DIR = TEST_OUTPUT_DIR / "screenshots"
ALERT_SCREENSHOTS_DIR = SCREENSHOTS_DIR / "alerts"


class TestUI(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    os.environ["SCALE"] = "1"
    sys.modules["mouseinfo"] = False

  @classmethod
  def tearDownClass(cls):
    del sys.modules["mouseinfo"]

  def setup(self, prime_type :PrimeType = PrimeType.NONE):
    set_prime(prime_type)
    self.sm = SubMaster(["uiDebug"])
    self.pm = PubMaster(["deviceState", "pandaStates", "controlsState", 'roadCameraState', 'wideRoadCameraState', 'liveLocationKalman', 'navInstruction', 'navRoute', 'modelV2'])
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
    #self.assertEqual(im.width, 2160)
    #self.assertEqual(im.height, 1080)
    img = np.array(im)
    im.close()
    return img

  def click(self, x, y, *args, **kwargs):
    import pyautogui
    pyautogui.click(self.ui.left + x, self.ui.top + y, *args, **kwargs)
    time.sleep(UI_DELAY) # give enough time for the UI to react

  @parameterized.expand(BASE_CASES)
  @with_processes(["ui"])
  def test_ui(self, name, setup_info, *args):
    if isinstance(setup_info, list):
      setup_case, prime_type = setup_info
    else:
      setup_case = setup_info
      prime_type = PrimeType.NONE
    self.setup(prime_type)

    setup_case(self.click, self.pm, *args)

    time.sleep(UI_DELAY) # wait a bit more for the UI to finish rendering

    im = self.screenshot()
    plt.imsave(SCREENSHOTS_DIR / f"{name}_PrimeType_{prime_type}.png", im)

  @with_processes(["ui"])
  def test_ui_events(self):
    cases = get_onroad_alert_cases()
    self.setup()
    setup_onroad(self.click, self.pm)
    time.sleep(UI_DELAY)
    for name, setup_case in cases.items():
      print(f'Testing Event {name}')
      setup_case(self.click, self.pm)
      im = self.screenshot()
      plt.imsave(ALERT_SCREENSHOTS_DIR / f"{name}.png", im)

  @with_processes(["ui"])
  def test_offroad_alerts(self):
    self.setup()
    setup_common(self.click, self.pm)
    time.sleep(UI_DELAY)
    for offroad_alert in OFFROAD_ALERTS.items():
      send_offroad_alert(self.click, self.pm, offroad_alert)
      im = self.screenshot()
      set_offroad_alert(offroad_alert[0], False)
      plt.imsave(ALERT_SCREENSHOTS_DIR / f"{offroad_alert[0]}.png", im)

def create_html_report():
  OUTPUT_FILE = TEST_OUTPUT_DIR / "base_index.html"

  with open(TEST_DIR / "template.html") as f:
    template = jinja2.Template(f.read())

  cases = {f.stem: (str(f.relative_to(TEST_OUTPUT_DIR)), "reference.png") for f in SCREENSHOTS_DIR.glob("*.png")}
  cases = dict(sorted(cases.items()))

  with open(OUTPUT_FILE, "w") as f:
    f.write(template.render(cases=cases))


  OUTPUT_FILE = TEST_OUTPUT_DIR / "alert_index.html"

  with open(TEST_DIR / "template.html") as f:
    template = jinja2.Template(f.read())

  cases = {f.stem: (str(f.relative_to(TEST_OUTPUT_DIR)), "reference.png") for f in ALERT_SCREENSHOTS_DIR.glob("*.png")}
  cases = dict(sorted(cases.items()))

  with open(OUTPUT_FILE, "w") as f:
    f.write(template.render(cases=cases))

def create_screenshots():
  if TEST_OUTPUT_DIR.exists():
    shutil.rmtree(TEST_OUTPUT_DIR)

  SCREENSHOTS_DIR.mkdir(parents=True)
  ALERT_SCREENSHOTS_DIR.mkdir(parents=True)
  unittest.main(exit=False)

if __name__ == "__main__":
  print("creating test screenshots")
  create_screenshots()

  print("creating html report")
  create_html_report()
