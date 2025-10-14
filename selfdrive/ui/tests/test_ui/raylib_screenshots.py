#!/usr/bin/env python3
import os
import sys
import shutil
import time
import pathlib
import pickle
from collections import namedtuple

import pyautogui
import pywinctl

from cereal import log
from cereal import messaging
from cereal.messaging import PubMaster, sub_sock
from msgq.visionipc import VisionIpcServer, VisionStreamType
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.test.helpers import with_processes
from openpilot.selfdrive.test.process_replay.migration import migrate, migrate_controlsState, migrate_carState
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from openpilot.system.updated.updated import parse_release_notes
from openpilot.tools.lib.cache import DEFAULT_CACHE_DIR
from openpilot.tools.lib.framereader import FrameReader
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route

TEST_DIR = pathlib.Path(__file__).parent
TEST_OUTPUT_DIR = TEST_DIR / "raylib_report"
SCREENSHOTS_DIR = TEST_OUTPUT_DIR / "screenshots"
UI_DELAY = 0.2

# Offroad alerts to test
OFFROAD_ALERTS = ["Offroad_IsTakingSnapshot"]

# Onroad test data
TEST_ROUTE = "a2a0ccea32023010|2023-07-27--13-01-19"
TEST_ROUTE_SEGMENT = 2
STREAMS: list[tuple] = []
DATA: dict[str, messaging.capnp._DynamicStructBuilder] = dict.fromkeys(
  [
    "carParams",
    "deviceState",
    "pandaStates",
    "controlsState",
    "selfdriveState",
    "liveCalibration",
    "modelV2",
    "radarState",
    "driverMonitoringState",
    "carState",
    "driverStateV2",
    "roadCameraState",
    "wideRoadCameraState",
    "driverCameraState",
  ],
  None,
)


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


def setup_pair_device(click, pm: PubMaster):
  click(1950, 800)


def setup_offroad_alert(click, pm: PubMaster):
  set_offroad_alert("Offroad_TemperatureTooHigh", True, extra_text="99C")
  set_offroad_alert("Offroad_ExcessiveActuation", True, extra_text="longitudinal")
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


def setup_onroad(click, pm: PubMaster):
  # Start a visionipc server to feed frames
  vipc_server = VisionIpcServer("camerad")
  for stream_type, cam, _ in STREAMS:
    vipc_server.create_buffers(stream_type, 5, cam.width, cam.height)
  vipc_server.start_listener()

  uidebug_received_cnt = 0
  packet_id = 0
  uidebug_sock = sub_sock("uiDebug")

  # Condition check for uiDebug processing
  check_uidebug = DATA["deviceState"].deviceState.started and not DATA["carParams"].carParams.notCar

  # Loop until some uiDebug messages or a few cycles
  while uidebug_received_cnt <= 20:
    for service, data in DATA.items():
      if data:
        pm.send(service, data)
        data.clear_write_flag()

    for stream_type, _, image in STREAMS:
      vipc_server.send(stream_type, image, packet_id, packet_id, packet_id)

    if check_uidebug:
      while uidebug_sock.receive(non_blocking=True):
        uidebug_received_cnt += 1
    else:
      uidebug_received_cnt += 1

    packet_id += 1
    time.sleep(0.05)


def setup_onroad_disengaged(click, pm: PubMaster):
  DATA["selfdriveState"].selfdriveState.enabled = False
  setup_onroad(click, pm)
  DATA["selfdriveState"].selfdriveState.enabled = True


def setup_onroad_override(click, pm: PubMaster):
  DATA["selfdriveState"].selfdriveState.state = log.SelfdriveState.OpenpilotState.overriding
  setup_onroad(click, pm)
  DATA["selfdriveState"].selfdriveState.state = log.SelfdriveState.OpenpilotState.enabled


def setup_onroad_wide(click, pm: PubMaster):
  # widecam show in when in experimental mode and going slow
  DATA["selfdriveState"].selfdriveState.experimentalMode = True
  DATA["carState"].carState.vEgo = 5
  setup_onroad(click, pm)


def setup_onroad_sidebar(click, pm: PubMaster):
  setup_onroad(click, pm)
  click(500, 500)
  setup_onroad(click, pm)


def setup_onroad_wide_sidebar(click, pm: PubMaster):
  setup_onroad_wide(click, pm)
  click(500, 500)
  setup_onroad_wide(click, pm)


def setup_driver_camera(click, pm: PubMaster):
  setup_settings(click, pm)
  click(1980, 620)  # preview driver camera button
  DATA["deviceState"].deviceState.started = False
  setup_onroad(click, pm)
  DATA["deviceState"].deviceState.started = True


CASES = {
  # "homescreen": setup_homescreen,
  # "settings_device": setup_settings,
  # "settings_network": setup_settings_network,
  # "settings_toggles": setup_settings_toggles,
  # "settings_software": setup_settings_software,
  # "settings_software_download": setup_settings_software_download,
  # "settings_software_release_notes": setup_settings_software_release_notes,
  # "settings_firehose": setup_settings_firehose,
  # "settings_developer": setup_settings_developer,
  # "keyboard": setup_keyboard,
  # "pair_device": setup_pair_device,
  # "offroad_alert": setup_offroad_alert,
  # "homescreen_update_available": setup_homescreen_update_available,
  # "confirmation_dialog": setup_confirmation_dialog,
  # "experimental_mode_description": setup_experimental_mode_description,
  # "onroad": setup_onroad,
  # "onroad_disengaged": setup_onroad_disengaged,
  # "onroad_override": setup_onroad_override,
  # "onroad_sidebar": setup_onroad_sidebar,
  "onroad_wide": setup_onroad_wide,
  "onroad_wide_sidebar": setup_onroad_wide_sidebar,
  "driver_camera": setup_driver_camera,
}


class TestUI:
  def __init__(self):
    os.environ["SCALE"] = os.getenv("SCALE", "1")
    sys.modules["mouseinfo"] = False

  def setup(self):
    ds = DATA["deviceState"]
    ds.deviceState.networkType = log.DeviceState.NetworkType.wifi
    ds.deviceState.lastAthenaPingTime = 0  # show "connect offline" instead of "connect error"
    self.pm = PubMaster(list(DATA.keys()))
    for _ in range(5):
      self.pm.send("deviceState", ds)
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


def get_cached_frames(route: Route, segnum: int):
  # Ensure cache directory exists
  os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
  frames_cache = f"{DEFAULT_CACHE_DIR}/test_ui_frames"
  # Load frames from cache if available
  if os.path.isfile(frames_cache):
    with open(frames_cache, "rb") as f:
      # Read frames from cache
      frames = pickle.load(f)
      road_img = frames[0]
      wide_road_img = frames[1]
      driver_img = frames[2]
  else:
    with open(frames_cache, "wb") as f:
      # No cached frames, read from route and cache them
      print("no cached frames, reading from route")
      road_img = FrameReader(route.camera_paths()[segnum], pix_fmt="nv12").get(0)
      wide_road_img = FrameReader(route.ecamera_paths()[segnum], pix_fmt="nv12").get(0)
      driver_img = FrameReader(route.dcamera_paths()[segnum], pix_fmt="nv12").get(0)
      pickle.dump([road_img, wide_road_img, driver_img], f)
  return road_img, wide_road_img, driver_img


def prepare_onroad_data():
  route = Route(TEST_ROUTE)

  # Prepare route data
  qpaths = route.qlog_paths()
  lr = LogReader(qpaths[TEST_ROUTE_SEGMENT])
  DATA["carParams"] = next((event.as_builder() for event in lr if event.which() == "carParams"), None)
  for event in migrate(lr, [migrate_controlsState, migrate_carState]):
    if event.which() in DATA:
      DATA[event.which()] = event.as_builder()
    if all(DATA.values()):
      break

  # Prepare camera frames
  cam = DEVICE_CAMERAS.get(("tici", "ar0231"))
  if cam:
    road_img, wide_road_img, driver_img = get_cached_frames(route, TEST_ROUTE_SEGMENT)
    STREAMS.append((VisionStreamType.VISION_STREAM_ROAD, cam.fcam, road_img.flatten().tobytes()))
    STREAMS.append((VisionStreamType.VISION_STREAM_WIDE_ROAD, cam.ecam, wide_road_img.flatten().tobytes()))
    STREAMS.append((VisionStreamType.VISION_STREAM_DRIVER, cam.dcam, driver_img.flatten().tobytes()))


def create_screenshots():
  if TEST_OUTPUT_DIR.exists():
    shutil.rmtree(TEST_OUTPUT_DIR)
  SCREENSHOTS_DIR.mkdir(parents=True)

  # Prepare onroad data (route + frames)
  prepare_onroad_data()

  t = TestUI()
  with OpenpilotPrefix():
    params = Params()
    params.put("DongleId", "123456789012345")
    for name, setup in CASES.items():
      t.test_ui(name, setup)


if __name__ == "__main__":
  create_screenshots()
