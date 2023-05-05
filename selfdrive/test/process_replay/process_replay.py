#!/usr/bin/env python3
import os
import time
import signal
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

from tqdm import tqdm

import cereal.messaging as messaging
from cereal import car, log
from cereal.services import service_list
from common.params import Params
from common.timeout import Timeout
from common.realtime import DT_CTRL
from panda.python import ALTERNATIVE_EXPERIENCE
from selfdrive.car.car_helpers import get_car, interfaces
from selfdrive.manager.process_config import managed_processes

# Numpy gives different results based on CPU features after version 19
NUMPY_TOLERANCE = 1e-7
CI = "CI" in os.environ
TIMEOUT = 15
PROC_REPLAY_DIR = os.path.dirname(os.path.abspath(__file__))
FAKEDATA = os.path.join(PROC_REPLAY_DIR, "fakedata/")


class ReplayContext:
  def __init__(self, cfg):
    self.non_polled_pubs = list(set(cfg.pub_sub.keys()) - set(cfg.polled_pubs))
    self.polled_pubs = cfg.polled_pubs
    self.drained_pubs = cfg.drained_pubs
    assert(len(self.non_polled_pubs) != 0 or len(self.polled_pubs) != 0)
    assert(set(self.drained_pubs) & set(self.polled_pubs) == set())
    self.subs = [s for sub in cfg.pub_sub.values() for s in sub]
  
  def __enter__(self):
    messaging.toggle_fake_events(True)

    self.recv_called_events = {
      s: messaging.fake_event(s, messaging.FAKE_EVENT_RECV_CALLED)
      for s in self.non_polled_pubs
    }
    self.recv_ready_events = {
      s: messaging.fake_event(s, messaging.FAKE_EVENT_RECV_READY)
      for s in self.non_polled_pubs
    }
    if len(self.polled_pubs) > 0 and len(self.drained_pubs) == 0:
      self.poll_called_event = messaging.fake_event("", messaging.FAKE_EVENT_POLL_CALLED)
      self.poll_ready_event = messaging.fake_event("", messaging.FAKE_EVENT_POLL_READY)

    return self

  def __exit__(self, exc_type, exc_obj, exc_tb):
    messaging.toggle_fake_events(False)

    del self.recv_called_events
    del self.recv_ready_events
    if len(self.polled_pubs) > 0 and len(self.drained_pubs) == 0:
      del self.poll_called_event
      del self.poll_ready_event

  def unlock_sockets(self, msg_type):
    if len(self.non_polled_pubs) <= 1 and len(self.polled_pubs) == 0:
      return
    
    if len(self.polled_pubs) > 0 and len(self.drained_pubs) == 0:
      self.poll_called_event.wait()
      self.poll_called_event.clear()
      if msg_type not in self.polled_pubs:
        self.poll_ready_event.set()

    for pub in self.non_polled_pubs:
      if pub == msg_type:
        continue

      self.recv_ready_events[pub].set()

  def wait_for_next_recv(self, msg_type, next_msg_type):
    if len(self.drained_pubs) > 0:
      return self._wait_for_next_recv_drained(msg_type, next_msg_type)
    elif len(self.polled_pubs) > 0:
      return self._wait_for_next_recv_using_polls(msg_type)
    else:
      return self._wait_for_next_recv_standard(msg_type)

  def _wait_for_next_recv_drained(self, msg_type, next_msg_type):
    # if the next message is also drained message, then we need to fake the recv_ready event
    # in order to start the next cycle
    if msg_type in self.drained_pubs and next_msg_type == msg_type:
      self.recv_called_events[self.drained_pubs[0]].wait()
      self.recv_called_events[self.drained_pubs[0]].clear()
      self.recv_ready_events[self.drained_pubs[0]].set()
    self.recv_called_events[self.drained_pubs[0]].wait()

  def _wait_for_next_recv_standard(self, msg_type):
    if len(self.non_polled_pubs) <= 1 and len(self.polled_pubs) == 0:
      return self.recv_called_events[self.non_polled_pubs[0]].wait()

    values = defaultdict(int)
    # expected sets is len(self.pubs) - 1 (current) + 1 (next)
    if len(self.non_polled_pubs) == 0:
      return

    # there're multiple sockets, and we don't know the order of recv calls,
    # so we wait for recv_called events manually
    # ugly and slow but works
    expected_total_sets = len(self.non_polled_pubs)
    while expected_total_sets > 0:
      for pub in self.non_polled_pubs:
        if not self.recv_called_events[pub].peek():
          continue

        value = self.recv_called_events[pub].clear()
        values[pub] += value
        expected_total_sets -= value
      time.sleep(0)

    max_key = max(values.keys(), key=lambda k: values[k])
    self.recv_called_events[max_key].set()

  def _wait_for_next_recv_using_polls(self, msg_type):
    if msg_type in self.polled_pubs:
      self.poll_ready_event.set()
    self.poll_called_event.wait()


@dataclass
class ProcessConfig:
  proc_name: str
  pub_sub: Dict[str, List[str]]
  ignore: List[str]
  init_callback: Optional[Callable]
  should_recv_callback: Optional[Callable]
  tolerance: Optional[float]
  environ: Dict[str, str] = field(default_factory=dict)
  subtest_name: str = ""
  field_tolerances: Dict[str, float] = field(default_factory=dict)
  timeout: int = 30
  polled_pubs: List[str] = field(default_factory=list)
  drained_pubs: List[str] = field(default_factory=list)


class DummySocket:
  def __init__(self):
    self.data = []

  def receive(self, non_blocking=False):
    if non_blocking:
      return None

    return self.data.pop()

  def send(self, data):
    self.data.append(data)


def fingerprint(rc, pm, msgs, fingerprint):
  print("start fingerprinting")

  canmsgs = [msg for msg in msgs if msg.which() == "can"]
  pm.send("pandaStates", messaging.new_message("pandaStates", 0))
  for m in canmsgs[:300]:
    pm.send("can", m.as_builder().to_bytes())
  rc.wait_for_next_recv("can", "can")
  print("complete fingerprinting")
  get_car_params(rc, pm, msgs, fingerprint)


def get_car_params(rc, pm, msgs, fingerprint):
  if fingerprint:
    CarInterface, _, _ = interfaces[fingerprint]
    CP = CarInterface.get_non_essential_params(fingerprint)
  else:
    can = DummySocket()
    sendcan = DummySocket()

    canmsgs = [msg for msg in msgs if msg.which() == 'can']
    for m in canmsgs[:300]:
      can.send(m.as_builder().to_bytes())
    _, CP = get_car(can, sendcan, Params().get_bool("ExperimentalLongitudinalEnabled"))
  Params().put("CarParams", CP.to_bytes())


def controlsd_rcv_callback(msg, CP, cfg, frame):
  # no sendcan until controlsd is initialized
  socks = [s for s in cfg.pub_sub[msg.which()] if
           (frame) % int(service_list[msg.which()].frequency / service_list[s].frequency) == 0]
  if "sendcan" in socks and (frame - 1) < 2000:
    socks.remove("sendcan")
  return socks, len(socks) > 0


def radar_rcv_callback(msg, CP, cfg, frame):
  if msg.which() != "can":
    return [], False
  elif CP.radarUnavailable:
    return ["radarState", "liveTracks"], True

  radar_msgs = {"honda": [0x445], "toyota": [0x19f, 0x22f], "gm": [0x474],
                "chrysler": [0x2d4]}.get(CP.carName, None)

  if radar_msgs is None:
    raise NotImplementedError

  for m in msg.can:
    if m.src == 1 and m.address in radar_msgs:
      return ["radarState", "liveTracks"], True
  return [], False


def calibration_rcv_callback(msg, CP, cfg, frame):
  # calibrationd publishes 1 calibrationData every 5 cameraOdometry packets.
  # should_recv always true to increment frame
  recv_socks = []
  if frame == 0 or (msg.which() == 'cameraOdometry' and (frame % 5) == 0):
    recv_socks = ["liveCalibration"]
  return recv_socks, (frame - 1) == 0 or msg.which() == 'cameraOdometry'


def torqued_rcv_callback(msg, CP, cfg, frame):
  # should_recv always true to increment frame
  recv_socks = []
  if msg.which() == 'liveLocationKalman' and (frame % 5) == 0:
    recv_socks = ["liveTorqueParameters"]
  return recv_socks, (frame - 1) == 0 or msg.which() == 'liveLocationKalman'


def ublox_rcv_callback(msg, CP, cfg, frame):
  msg_class, msg_id = msg.ubloxRaw[2:4]
  if (msg_class, msg_id) in {(1, 7 * 16)}:
    return ["gpsLocationExternal"], True
  elif (msg_class, msg_id) in {(2, 1 * 16 + 5), (10, 9)}:
    return ["ubloxGnss"], True
  else:
    return [], False


def rate_based_rcv_callback(msg, CP, cfg, frame):
  resp_sockets = [s for s in cfg.pub_sub[msg.which()] if
          frame % max(1, int(service_list[msg.which()].frequency / service_list[s].frequency)) == 0]
  should_recv = bool(len(resp_sockets))

  return resp_sockets, should_recv


CONFIGS = [
  ProcessConfig(
    proc_name="controlsd",
    pub_sub={
      "can": ["controlsState", "carState", "carControl", "sendcan", "carEvents", "carParams"],
      "deviceState": [], "pandaStates": [], "peripheralState": [], "liveCalibration": [], "driverMonitoringState": [],
      "longitudinalPlan": [], "lateralPlan": [], "liveLocationKalman": [], "liveParameters": [], "radarState": [],
      "modelV2": [], "driverCameraState": [], "roadCameraState": [], "wideRoadCameraState": [], "managerState": [],
      "testJoystick": [], "liveTorqueParameters": [],
    },
    ignore=["logMonoTime", "valid", "controlsState.startMonoTime", "controlsState.cumLagMs"],
    init_callback=fingerprint,
    should_recv_callback=controlsd_rcv_callback,
    tolerance=NUMPY_TOLERANCE,
    polled_pubs=[
      "deviceState", "pandaStates", "peripheralState", "liveCalibration", "driverMonitoringState",
      "longitudinalPlan", "lateralPlan", "liveLocationKalman", "liveParameters", "radarState",
      "modelV2", "driverCameraState", "roadCameraState", "wideRoadCameraState", "managerState",
      "testJoystick", "liveTorqueParameters"
    ],
    drained_pubs=["can"]
  ),
  ProcessConfig(
    proc_name="radard",
    pub_sub={
      "can": ["radarState", "liveTracks"],
      "carState": [], "modelV2": [],
    },
    ignore=["logMonoTime", "valid", "radarState.cumLagMs"],
    init_callback=get_car_params,
    should_recv_callback=radar_rcv_callback,
    tolerance=None,
    polled_pubs=["carState", "modelV2"],
    drained_pubs=["can"]
  ),
  ProcessConfig(
    proc_name="plannerd",
    pub_sub={
      "modelV2": ["lateralPlan", "longitudinalPlan", "uiPlan"],
      "carControl": [], "carState": [], "controlsState": [], "radarState": [],
    },
    ignore=["logMonoTime", "valid", "longitudinalPlan.processingDelay", "longitudinalPlan.solverExecutionTime", "lateralPlan.solverExecutionTime"],
    init_callback=get_car_params,
    should_recv_callback=rate_based_rcv_callback,
    tolerance=NUMPY_TOLERANCE,
    polled_pubs=["radarState", "modelV2"],
  ),
  ProcessConfig(
    proc_name="calibrationd",
    pub_sub={
      "carState": ["liveCalibration"],
      "cameraOdometry": [],
      "carParams": [],
    },
    ignore=["logMonoTime", "valid"],
    init_callback=get_car_params,
    should_recv_callback=calibration_rcv_callback,
    tolerance=None,
    polled_pubs=["cameraOdometry"],
  ),
  ProcessConfig(
    proc_name="dmonitoringd",
    pub_sub={
      "driverStateV2": ["driverMonitoringState"],
      "liveCalibration": [], "carState": [], "modelV2": [], "controlsState": [],
    },
    ignore=["logMonoTime", "valid"],
    init_callback=get_car_params,
    should_recv_callback=rate_based_rcv_callback,
    tolerance=NUMPY_TOLERANCE,
    polled_pubs=["driverStateV2"],
  ),
  ProcessConfig(
    proc_name="locationd",
    pub_sub={
      "cameraOdometry": ["liveLocationKalman"],
      "accelerometer": [], "gyroscope": [],
      "gpsLocationExternal": [], "liveCalibration": [], 
      "carState": [], "carParams": [], "gpsLocation": [],
    },
    ignore=["logMonoTime", "valid"],
    init_callback=get_car_params,
    should_recv_callback=None,
    tolerance=NUMPY_TOLERANCE,
    polled_pubs=[
      "cameraOdometry", "gpsLocationExternal", "gyroscope",
      "accelerometer", "liveCalibration", "carState", "carParams", "gpsLocation"
    ],
  ),
  ProcessConfig(
    proc_name="paramsd",
    pub_sub={
      "liveLocationKalman": ["liveParameters"],
      "carState": []
    },
    ignore=["logMonoTime", "valid"],
    init_callback=get_car_params,
    should_recv_callback=rate_based_rcv_callback,
    tolerance=NUMPY_TOLERANCE,
    polled_pubs=["liveLocationKalman"],
  ),
  ProcessConfig(
    proc_name="ubloxd",
    pub_sub={
      "ubloxRaw": ["ubloxGnss", "gpsLocationExternal"],
    },
    ignore=["logMonoTime"],
    init_callback=None,
    should_recv_callback=None,
    tolerance=None,
  ),
  ProcessConfig(
    proc_name="laikad",
    pub_sub={
      "ubloxGnss": ["gnssMeasurements"],
      "qcomGnss": ["gnssMeasurements"],
    },
    ignore=["logMonoTime"],
    init_callback=get_car_params,
    should_recv_callback=None,
    tolerance=NUMPY_TOLERANCE,
    timeout=60*10,  # first messages are blocked on internet assistance
    drained_pubs=["ubloxGnss", "qcomGnss"],
  ),
  ProcessConfig(
    proc_name="torqued",
    pub_sub={
      "liveLocationKalman": ["liveTorqueParameters"],
      "carState": [], "carControl": [],
    },
    ignore=["logMonoTime"],
    init_callback=get_car_params,
    should_recv_callback=torqued_rcv_callback,
    tolerance=NUMPY_TOLERANCE,
    polled_pubs=["liveLocationKalman"],
  ),
]


def replay_process(cfg, lr, fingerprint=None):
  with ReplayContext(cfg) as rc:
    pm = messaging.PubMaster(cfg.pub_sub.keys())
    sub_sockets = [s for _, sub in cfg.pub_sub.items() for s in sub]
    sockets = {s: messaging.sub_sock(s, timeout=100) for s in sub_sockets}

    all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)
    pub_msgs = [msg for msg in all_msgs if msg.which() in list(cfg.pub_sub.keys())]

    # We need to fake SubMaster alive since we can't inject a fake clock
    setup_env(simulation=True, cfg=cfg, lr=lr)

    if cfg.proc_name == "laikad":
      ublox = Params().get_bool("UbloxAvailable")
      sub_keys = ({"qcomGnss", } if ublox else {"ubloxGnss", })
      keys = set(rc.non_polled_pubs) - sub_keys
      d_keys = set(rc.drained_pubs) - sub_keys
      rc.non_polled_pubs = list(keys)
      rc.drained_pubs = list(d_keys)
      pub_msgs = [msg for msg in pub_msgs if msg.which() in keys]

    if cfg.proc_name == "locationd":
      ublox = Params().get_bool("UbloxAvailable")
      sub_keys = ({"gpsLocation", } if ublox else {"gpsLocationExternal", })
      np_keys = set(rc.non_polled_pubs) - sub_keys
      p_keys = set(rc.polled_pubs) - sub_keys
      keys = np_keys | p_keys
      rc.non_polled_pubs = list(np_keys)
      rc.polled_pubs = list(p_keys)
      pub_msgs = [msg for msg in pub_msgs if msg.which() in keys]

    controlsState = None
    initialized = False
    for msg in lr:
      if msg.which() == 'controlsState':
        controlsState = msg.controlsState
        if initialized:
          break
      elif msg.which() == 'carEvents':
        initialized = car.CarEvent.EventName.controlsInitializing not in [e.name for e in msg.carEvents]

    assert controlsState is not None and initialized, "controlsState never initialized"

    if fingerprint is not None:
      os.environ['SKIP_FW_QUERY'] = "1"
      os.environ['FINGERPRINT'] = fingerprint
      setup_env(cfg=cfg, controlsState=controlsState, lr=lr)
    else:
      CP = [m for m in lr if m.which() == 'carParams'][0].carParams
      setup_env(CP=CP, cfg=cfg, controlsState=controlsState, lr=lr)

    managed_processes[cfg.proc_name].prepare()
    managed_processes[cfg.proc_name].start()

    if cfg.init_callback is not None:
      cfg.init_callback(rc, pm, all_msgs, fingerprint)
      CP = car.CarParams.from_bytes(Params().get("CarParams", block=True))

    log_msgs = []
    try:
      # Wait for process to startup
      with Timeout(10, error_msg=f"timed out waiting for process to start: {repr(cfg.proc_name)}"):
        while not any(pm.all_readers_updated(s) for s in cfg.pub_sub.keys()):
          time.sleep(0)
      
      for s in sockets.values():
        messaging.recv_one_or_none(s)

      # Do the replay
      cnt = 0
      for i, msg in enumerate(tqdm(pub_msgs)):
        with Timeout(cfg.timeout, error_msg=f"timed out testing process {repr(cfg.proc_name)}, {cnt}/{len(pub_msgs)} msgs done"):
          resp_sockets = cfg.pub_sub[msg.which()]
          should_recv = True
          if cfg.should_recv_callback is not None:
            resp_sockets, should_recv = cfg.should_recv_callback(msg, CP, cfg, cnt)

          if len(log_msgs) == 0 and len(resp_sockets) > 0:
            for s in sockets.values():
              messaging.recv_one_or_none(s)
          
          rc.unlock_sockets(msg.which())
          pm.send(msg.which(), msg.as_builder())
          # wait for the next receive on the process side
          rc.wait_for_next_recv(msg.which(), pub_msgs[i+1].which() if i+1 < len(pub_msgs) else None)

          if should_recv:
            for s in resp_sockets:
              ms = messaging.drain_sock(sockets[s])
              for m in ms:
                m = m.as_builder()
                m.logMonoTime = msg.logMonoTime
                log_msgs.append(m.as_reader())
            cnt += 1
    finally:
      managed_processes[cfg.proc_name].signal(signal.SIGKILL)
      managed_processes[cfg.proc_name].stop()

    return log_msgs


def setup_env(simulation=False, CP=None, cfg=None, controlsState=None, lr=None):
  params = Params()
  params.clear_all()
  params.put_bool("OpenpilotEnabledToggle", True)
  params.put_bool("Passive", False)
  params.put_bool("DisengageOnAccelerator", True)
  params.put_bool("WideCameraOnly", False)
  params.put_bool("DisableLogging", False)

  os.environ["NO_RADAR_SLEEP"] = "1"
  os.environ["REPLAY"] = "1"
  os.environ["SKIP_FW_QUERY"] = ""
  os.environ["FINGERPRINT"] = ""

  if lr is not None:
    services = {m.which() for m in lr}
    params.put_bool("UbloxAvailable", "ubloxGnss" in services)
  
  if cfg is not None:
    # Clear all custom processConfig environment variables
    for config in CONFIGS:
      for k, _ in config.environ.items():
        if k in os.environ:
          del os.environ[k]

    os.environ.update(cfg.environ)
    os.environ['PROC_NAME'] = cfg.proc_name

  if simulation:
    os.environ["SIMULATION"] = "1"
  elif "SIMULATION" in os.environ:
    del os.environ["SIMULATION"]

  # Initialize controlsd with a controlsState packet
  if controlsState is not None:
    params.put("ReplayControlsState", controlsState.as_builder().to_bytes())
  else:
    params.remove("ReplayControlsState")

  # Regen or python process
  if CP is not None:
    if CP.alternativeExperience == ALTERNATIVE_EXPERIENCE.DISABLE_DISENGAGE_ON_GAS:
      params.put_bool("DisengageOnAccelerator", False)

    if CP.fingerprintSource == "fw":
      params.put("CarParamsCache", CP.as_builder().to_bytes())
    else:
      os.environ['SKIP_FW_QUERY'] = "1"
      os.environ['FINGERPRINT'] = CP.carFingerprint

    if CP.openpilotLongitudinalControl:
      params.put_bool("ExperimentalLongitudinalEnabled", True)


def check_enabled(msgs):
  cur_enabled_count = 0
  max_enabled_count = 0
  for msg in msgs:
    if msg.which() == "carParams":
      if msg.carParams.notCar:
        return True
    elif msg.which() == "controlsState":
      if msg.controlsState.active:
        cur_enabled_count += 1
      else:
        cur_enabled_count = 0
      max_enabled_count = max(max_enabled_count, cur_enabled_count)

  return max_enabled_count > int(10. / DT_CTRL)
