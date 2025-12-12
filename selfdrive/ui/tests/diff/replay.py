#!/usr/bin/env python3
import os
import time
import coverage
import pyray as rl
import pickle
from dataclasses import dataclass
from openpilot.selfdrive.ui.tests.diff.diff import DIFF_OUT_DIR
from openpilot.system.hardware import HARDWARE

HARDWARE.get_device_type = lambda: "mici"

os.environ["RECORD"] = "1"
if "RECORD_OUTPUT" not in os.environ:
  os.environ["RECORD_OUTPUT"] = "mici_ui_replay.mp4"

os.environ["RECORD_OUTPUT"] = os.path.join(DIFF_OUT_DIR, os.environ["RECORD_OUTPUT"])

from cereal import messaging, log
from msgq.visionipc import VisionIpcServer, VisionStreamType
from openpilot.common.params import Params
from openpilot.system.version import terms_version, training_version
from openpilot.system.ui.lib.application import gui_app, MousePos, MouseEvent
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout
from openpilot.tools.plotjuggler.juggle import DEMO_ROUTE
from openpilot.common.transformations.camera import CameraConfig, DEVICE_CAMERAS
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.framereader import FrameReader
from openpilot.tools.lib.route import Route
from openpilot.tools.lib.cache import DEFAULT_CACHE_DIR
from openpilot.selfdrive.selfdrived.events import EVENTS, ET, Events
from openpilot.selfdrive.selfdrived.alertmanager import AlertManager, set_offroad_alert

EventName = log.OnroadEvent.EventName
SelfdriveState = log.SelfdriveState

FPS = 60
HEADLESS = os.getenv("WINDOWED", "0") == "1"

STREAMS: list[tuple[VisionStreamType, CameraConfig, bytes]] = []

alert_cycle_start_frame = None
ALERTS = [EventName.preLaneChangeRight, EventName.steerSaturated, EventName.steerUnavailable]


def _update_state():
  """We manually update state"""


ui_state._update_state = _update_state


class FakeSubMaster:
  def __init__(self, names):
    self.data = {name: getattr(messaging.new_message(name), name) for name in names}

  def __getitem__(self, name):
    return self.data[name]


@dataclass
class DummyEvent:
  click: bool = False
  # TODO: add some kind of intensity
  swipe_left: bool = False
  swipe_right: bool = False
  swipe_down: bool = False
  onroad: bool = False
  cycle_alerts: bool = False
  offroad: bool = False


SCRIPT = [
  (0, DummyEvent(onroad=True)),
  (FPS * 1, DummyEvent(click=True)),
  (FPS * 2, DummyEvent(click=True)),
  (FPS * 3, DummyEvent(cycle_alerts=True)),
  # (FPS * 3, DummyEvent(swipe_down=True)),
  # (FPS * 4, DummyEvent(swipe_down=True)),
  # (FPS * 5, DummyEvent(swipe_left=True, onroad=True)),
  (FPS * 10, DummyEvent()),
]


def setup_state():
  params = Params()
  params.put("HasAcceptedTerms", terms_version)
  params.put("CompletedTrainingVersion", training_version)
  params.put("DongleId", "test123456789")
  params.put("UpdaterCurrentDescription", "0.10.1 / test-branch / abc1234 / Nov 30")


def setup_visionipc():
  cam = DEVICE_CAMERAS[("tici", "ar0231")]

  route = Route(DEMO_ROUTE)
  segnum = 2
  # lr = LogReader(route.qlog_paths()[segnum])

  frames_cache = f'{DEFAULT_CACHE_DIR}/ui_frames'
  if os.path.isfile(frames_cache):
    with open(frames_cache, 'rb') as f:
      frames = pickle.load(f)
      road_img = frames[0]
      wide_road_img = frames[1]
      driver_img = frames[2]
  else:
    with open(frames_cache, 'wb') as f:
      road_img = FrameReader(route.camera_paths()[segnum], pix_fmt="nv12").get(0)
      wide_road_img = FrameReader(route.ecamera_paths()[segnum], pix_fmt="nv12").get(0)
      driver_img = FrameReader(route.dcamera_paths()[segnum], pix_fmt="nv12").get(0)
      pickle.dump([road_img, wide_road_img, driver_img], f)

  STREAMS.append((VisionStreamType.VISION_STREAM_ROAD, cam.fcam, road_img.flatten().tobytes()))
  STREAMS.append((VisionStreamType.VISION_STREAM_WIDE_ROAD, cam.ecam, wide_road_img.flatten().tobytes()))
  STREAMS.append((VisionStreamType.VISION_STREAM_DRIVER, cam.dcam, driver_img.flatten().tobytes()))

  vipc_server = VisionIpcServer("camerad")
  for stream_type, cam, _ in STREAMS:
    vipc_server.create_buffers(stream_type, 5, cam.width, cam.height)
  vipc_server.start_listener()
  return vipc_server


def send_vipc_frame(vipc_server: VisionIpcServer, packet_id: int):
  for stream_type, _, image in STREAMS:
    vipc_server.send(stream_type, image, packet_id, packet_id, packet_id)
  packet_id += 1


def inject_click(coords):
  events = []
  x, y = coords[0]
  events.append(MouseEvent(pos=MousePos(x, y), slot=0, left_pressed=True, left_released=False, left_down=False, t=time.monotonic()))
  for x, y in coords[1:]:
    events.append(MouseEvent(pos=MousePos(x, y), slot=0, left_pressed=False, left_released=False, left_down=True, t=time.monotonic()))
  x, y = coords[-1]
  events.append(MouseEvent(pos=MousePos(x, y), slot=0, left_pressed=False, left_released=True, left_down=False, t=time.monotonic()))

  with gui_app._mouse._lock:
    gui_app._mouse._events.extend(events)


def handle_event(event: DummyEvent, pm: messaging.PubMaster, frame: int):
  global alert_cycle_start_frame

  if event.click:
    inject_click([(gui_app.width // 2, gui_app.height // 2)])
  if event.swipe_left:
    inject_click([(gui_app.width * 3 // 4, gui_app.height // 2),
                  (gui_app.width // 4, gui_app.height // 2),
                  (0, gui_app.height // 2)])
  if event.swipe_right:
    inject_click([(gui_app.width // 4, gui_app.height // 2),
                  (gui_app.width * 3 // 4, gui_app.height // 2),
                  (gui_app.width, gui_app.height // 2)])
  if event.swipe_down:
    inject_click([(gui_app.width // 2, gui_app.height // 4),
                  (gui_app.width // 2, gui_app.height * 3 // 4),
                  (gui_app.width // 2, gui_app.height)])
  if event.onroad:
    ui_state.started = True
  if event.offroad:
    ui_state.started = False
  if event.cycle_alerts:
    alert_cycle_start_frame = frame


def cycle_alerts_step(pm: messaging.PubMaster, frame: int):
  # alerts = [
  #   ("High Beam On", "Turn off high beam", 5.0),
  #   ("Speed Limit Ahead", "Reduce speed to 45 mph", 5.0),
  #   ("Steering Required", "Please steer the vehicle", 5.0),
  #   ("System Malfunction", "Contact support", 5.0),
  # ]

  global alert_cycle_start_frame
  if alert_cycle_start_frame is None:
    return
  elapsed_frames = frame - alert_cycle_start_frame
  alert_index = (elapsed_frames // (FPS * 2)) % (len(ALERTS) + 1)

  event = ALERTS[alert_index] if alert_index < len(ALERTS) else None
  if event is None:
    alert_cycle_start_frame = None
    return

  alert = None
  for et, _alert in EVENTS[event].items():
    alert = _alert
    break

  assert alert is not None, f"Alert data for {event} not found"

  sm = FakeSubMaster(['carControl'])
  sm['carControl'].actuators.torque = 1.0

  ev = Events()
  ev.add(event)
  alerts = ev.create_alerts([ET.WARNING, ET.PERMANENT, ET.SOFT_DISABLE], [None, None, sm, None, 1.0, None])

  if callable(alert):
    alert = alert(None, None, sm, None, 1.0, None)

  AM = AlertManager()
  AM.add_many(frame, alerts)
  AM.process_alerts(frame, set())
  print('AM.current_alert', AM.current_alert)

  # for et in ET:
  #   if et in EVENTS[event]:
  #     alert = EVENTS[event][et]
  #     break

  msg = messaging.new_message("selfdriveState")
  ss = msg.selfdriveState
  ss.alertText1 = AM.current_alert.alert_text_1
  ss.alertText2 = AM.current_alert.alert_text_2
  ss.alertSize = AM.current_alert.alert_size
  ss.alertStatus = AM.current_alert.alert_status
  ss.alertType = AM.current_alert.alert_type

  pm.send("selfdriveState", msg)


def run_replay():
  setup_state()
  vipc_server = setup_visionipc()
  os.makedirs(DIFF_OUT_DIR, exist_ok=True)

  pm = messaging.PubMaster([
    "selfdriveState",
  ])

  if HEADLESS:
    rl.set_config_flags(rl.FLAG_WINDOW_HIDDEN)
  gui_app.init_window("ui diff test", fps=FPS)
  main_layout = MiciMainLayout()
  main_layout.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))

  frame = 0
  script_index = 0

  for should_render in gui_app.render():
    send_vipc_frame(vipc_server, frame)
    while script_index < len(SCRIPT) and SCRIPT[script_index][0] == frame:
      _, event = SCRIPT[script_index]
      handle_event(event, pm, frame)
      script_index += 1

    if alert_cycle_start_frame is not None:
      cycle_alerts_step(pm, frame)

    ui_state.update()

    if should_render:
      main_layout.render()

    frame += 1

    if script_index >= len(SCRIPT):
      break

  gui_app.close()

  print(f"Total frames: {frame}")
  print(f"Video saved to: {os.environ['RECORD_OUTPUT']}")


def main():
  cov = coverage.coverage(source=['openpilot.selfdrive.ui.mici'])
  with cov.collect():
    run_replay()
  cov.stop()
  cov.save()
  cov.report()
  cov.html_report(directory=os.path.join(DIFF_OUT_DIR, 'htmlcov'))
  print("HTML report: htmlcov/index.html")


if __name__ == "__main__":
  main()
