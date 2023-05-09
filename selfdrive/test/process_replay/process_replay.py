#!/usr/bin/env python3
import os
import time
import signal
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

from tqdm import tqdm

import cereal.messaging as messaging
from cereal import car
from cereal.services import service_list
from common.params import Params
from common.timeout import Timeout
from common.realtime import DT_CTRL
from panda.python import ALTERNATIVE_EXPERIENCE
from selfdrive.car.car_helpers import get_car, interfaces
from selfdrive.manager.process_config import managed_processes
from selfdrive.test.process_replay.helpers import OpenpilotPrefix

# Numpy gives different results based on CPU features after version 19
NUMPY_TOLERANCE = 1e-7
CI = "CI" in os.environ
TIMEOUT = 15
PROC_REPLAY_DIR = os.path.dirname(os.path.abspath(__file__))
FAKEDATA = os.path.join(PROC_REPLAY_DIR, "fakedata/")


class ReplayContext:
  def __init__(self, cfg):
    self.non_polled_pubs = list(set(cfg.pubs) - set(cfg.polled_pubs))
    self.polled_pubs = cfg.polled_pubs
    self.drained_pub = cfg.drained_pub
    assert(len(self.non_polled_pubs) != 0 or len(self.polled_pubs) != 0 or self.drained_pub is not None)
  
  def __enter__(self):
    messaging.toggle_fake_events(True)

    pubs_with_events = self.non_polled_pubs if self.drained_pub is None else [self.drained_pub]
    self.recv_called_events = {
      s: messaging.fake_event(s, messaging.FAKE_EVENT_RECV_CALLED)
      for s in pubs_with_events
    }
    self.recv_ready_events = {
      s: messaging.fake_event(s, messaging.FAKE_EVENT_RECV_READY)
      for s in pubs_with_events
    }
    if len(self.polled_pubs) > 0 and self.drained_pub is None:
      self.poll_called_event = messaging.fake_event("", messaging.FAKE_EVENT_POLL_CALLED)
      self.poll_ready_event = messaging.fake_event("", messaging.FAKE_EVENT_POLL_READY)

    return self

  def __exit__(self, exc_type, exc_obj, exc_tb):
    del self.recv_called_events
    del self.recv_ready_events
    if len(self.polled_pubs) > 0 and self.drained_pub is None:
      del self.poll_called_event
      del self.poll_ready_event

    messaging.toggle_fake_events(False)

  def unlock_sockets(self, msg_type):
    if self.drained_pub is not None:
      return
    
    if len(self.polled_pubs) > 0:
      self.poll_called_event.wait()
      self.poll_called_event.clear()
      if msg_type not in self.polled_pubs:
        self.poll_ready_event.set()

    for pub in self.non_polled_pubs:
      if pub == msg_type:
        continue

      self.recv_ready_events[pub].set()

  def wait_for_next_recv(self, msg_type, should_recv):
    if self.drained_pub is not None:
      return self._wait_for_next_recv_drained(msg_type, should_recv)
    elif len(self.polled_pubs) > 0:
      return self._wait_for_next_recv_using_polls(msg_type)
    else:
      return self._wait_for_next_recv_standard(msg_type)

  def _wait_for_next_recv_drained(self, msg_type, should_recv):
    # if the next message is also drained message, then we need to fake the recv_ready event
    # in order to start the next cycle
    if not should_recv:
      return

    if msg_type == self.drained_pub:
      self.recv_called_events[self.drained_pub].wait()
      self.recv_called_events[self.drained_pub].clear()
      self.recv_ready_events[self.drained_pub].set()
    self.recv_called_events[self.drained_pub].wait()

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
  pubs: List[str]
  subs: List[str]
  ignore: List[str]
  config_callback: Optional[Callable]
  init_callback: Optional[Callable]
  should_recv_callback: Optional[Callable]
  tolerance: Optional[float]
  environ: Dict[str, str] = field(default_factory=dict)
  subtest_name: str = ""
  field_tolerances: Dict[str, float] = field(default_factory=dict)
  timeout: int = 30
  simulation: bool = True
  polled_pubs: List[str] = field(default_factory=list)
  drained_pub: Optional[str] = None


class DummySocket:
  def __init__(self):
    self.data = []

  def receive(self, non_blocking=False):
    if non_blocking:
      return None

    return self.data.pop()

  def send(self, data):
    self.data.append(data)


def controlsd_fingerprint_callback(rc, pm, msgs, fingerprint):
  print("start fingerprinting")
  params = Params()
  canmsgs = [msg for msg in msgs if msg.which() == "can"][:300]
  # controlsd expects one arbitrary can and pandaState
  pm.send("can", messaging.new_message("can", 1))
  pm.send("pandaStates", messaging.new_message("pandaStates", 1))
  pm.send("can", messaging.new_message("can", 1))
  rc.wait_for_next_recv("can", True)

  # fingerprinting is done, when CarParams is set
  while params.get("CarParams") is None:
    if len(canmsgs) == 0:
      raise ValueError("Fingerprinting failed. Run out of can msgs")

    m = canmsgs.pop(0)
    pm.send("can", m.as_builder().to_bytes())
    rc.wait_for_next_recv(None, True)


def get_car_params_callback(rc, pm, msgs, fingerprint):
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
  if msg.which() != "can":
    return [], False

  socks = [
    s for s in cfg.subs if
    (frame) % int(service_list[msg.which()].frequency / service_list[s].frequency) == 0
  ]
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
  resp_sockets = [
    s for s in cfg.subs 
    if frame % max(1, int(service_list[msg.which()].frequency / service_list[s].frequency)) == 0
  ]
  should_recv = bool(len(resp_sockets))

  return resp_sockets, should_recv


def laikad_config_pubsub_callback(params, cfg):
  ublox = params.get_bool("UbloxAvailable")
  drained_key = "ubloxGnss" if ublox else "qcomGnss"
  sub_keys = ({"qcomGnss", } if ublox else {"ubloxGnss", })

  return set(cfg.pubs) - sub_keys, drained_key


def locationd_config_pubsub_callback(params, cfg):
  ublox = params.get_bool("UbloxAvailable")
  sub_keys = ({"gpsLocation", } if ublox else {"gpsLocationExternal", })
  
  return set(cfg.pubs) - sub_keys, None


CONFIGS = [
  ProcessConfig(
    proc_name="controlsd",
    pubs=[
      "can", "deviceState", "pandaStates", "peripheralState", "liveCalibration", "driverMonitoringState",
      "longitudinalPlan", "lateralPlan", "liveLocationKalman", "liveParameters", "radarState",
      "modelV2", "driverCameraState", "roadCameraState", "wideRoadCameraState", "managerState",
      "testJoystick", "liveTorqueParameters"
    ],
    subs=["controlsState", "carState", "carControl", "sendcan", "carEvents", "carParams"],
    ignore=["logMonoTime", "valid", "controlsState.startMonoTime", "controlsState.cumLagMs"],
    config_callback=None,
    init_callback=controlsd_fingerprint_callback,
    should_recv_callback=controlsd_rcv_callback,
    tolerance=NUMPY_TOLERANCE,
    simulation=False,
    drained_pub="can",
  ),
  ProcessConfig(
    proc_name="radard",
    pubs=["can", "carState", "modelV2"],
    subs=["radarState", "liveTracks"],
    ignore=["logMonoTime", "valid", "radarState.cumLagMs"],
    config_callback=None,
    init_callback=get_car_params_callback,
    should_recv_callback=radar_rcv_callback,
    tolerance=None,
    drained_pub="can",
  ),
  ProcessConfig(
    proc_name="plannerd",
    pubs=["modelV2", "carControl", "carState", "controlsState", "radarState"],
    subs=["lateralPlan", "longitudinalPlan", "uiPlan"],
    ignore=["logMonoTime", "valid", "longitudinalPlan.processingDelay", "longitudinalPlan.solverExecutionTime", "lateralPlan.solverExecutionTime"],
    config_callback=None,
    init_callback=get_car_params_callback,
    should_recv_callback=rate_based_rcv_callback,
    tolerance=NUMPY_TOLERANCE,
    polled_pubs=["radarState", "modelV2"],
  ),
  ProcessConfig(
    proc_name="calibrationd",
    pubs=["carState", "cameraOdometry", "carParams"],
    subs=["liveCalibration"],
    ignore=["logMonoTime", "valid"],
    config_callback=None,
    init_callback=get_car_params_callback,
    should_recv_callback=calibration_rcv_callback,
    tolerance=None,
    polled_pubs=["cameraOdometry"],
  ),
  ProcessConfig(
    proc_name="dmonitoringd",
    pubs=["driverStateV2", "liveCalibration", "carState", "modelV2", "controlsState"],
    subs=["driverMonitoringState"],
    ignore=["logMonoTime", "valid"],
    config_callback=None,
    init_callback=get_car_params_callback,
    should_recv_callback=rate_based_rcv_callback,
    tolerance=NUMPY_TOLERANCE,
    polled_pubs=["driverStateV2"],
  ),
  ProcessConfig(
    proc_name="locationd",
    pubs=[
      "cameraOdometry", "accelerometer", "gyroscope", "gpsLocationExternal", 
      "liveCalibration", "carState", "carParams", "gpsLocation"
    ],
    subs=["liveLocationKalman"],
    ignore=["logMonoTime", "valid"],
    config_callback=locationd_config_pubsub_callback,
    init_callback=get_car_params_callback,
    should_recv_callback=None,
    tolerance=NUMPY_TOLERANCE,
    polled_pubs=[
      "cameraOdometry", "gpsLocationExternal", "gyroscope",
      "accelerometer", "liveCalibration", "carState", "carParams", "gpsLocation"
    ],
  ),
  ProcessConfig(
    proc_name="paramsd",
    pubs=["liveLocationKalman", "carState"],
    subs=["liveParameters"],
    ignore=["logMonoTime", "valid"],
    config_callback=None,
    init_callback=get_car_params_callback,
    should_recv_callback=rate_based_rcv_callback,
    tolerance=NUMPY_TOLERANCE,
    polled_pubs=["liveLocationKalman"],
  ),
  ProcessConfig(
    proc_name="ubloxd",
    pubs=["ubloxRaw"],
    subs=["ubloxGnss", "gpsLocationExternal"],
    ignore=["logMonoTime"],
    config_callback=None,
    init_callback=None,
    should_recv_callback=None,
    tolerance=None,
  ),
  ProcessConfig(
    proc_name="laikad",
    pubs=["ubloxGnss", "qcomGnss"],
    subs=["gnssMeasurements"],
    ignore=["logMonoTime"],
    config_callback=laikad_config_pubsub_callback,
    init_callback=get_car_params_callback,
    should_recv_callback=None,
    tolerance=NUMPY_TOLERANCE,
    timeout=60*10,  # first messages are blocked on internet assistance
    drained_pub="ubloxGnss", # config_callback will switch this to qcom if needed 
  ),
  ProcessConfig(
    proc_name="torqued",
    pubs=["liveLocationKalman", "carState", "carControl"],
    subs=["liveTorqueParameters"],
    ignore=["logMonoTime"],
    config_callback=None,
    init_callback=get_car_params_callback,
    should_recv_callback=torqued_rcv_callback,
    tolerance=NUMPY_TOLERANCE,
    polled_pubs=["liveLocationKalman"],
  ),
]


def replay_process(cfg, lr, fingerprint=None):
  with OpenpilotPrefix():
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

    CP = [m for m in lr if m.which() == 'carParams'][0].carParams
    if fingerprint is not None:
      setup_env(cfg=cfg, controlsState=controlsState, lr=lr, fingerprint=fingerprint)
    else:
      setup_env(CP=CP, cfg=cfg, controlsState=controlsState, lr=lr)

    if cfg.config_callback is not None:
      params = Params()
      cfg.pubs, cfg.drained_pub = cfg.config_callback(params, cfg)
      cfg.polled_pubs = list(set(cfg.polled_pubs) & set(cfg.pubs))

    all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)
    pub_msgs = [msg for msg in all_msgs if msg.which() in set(cfg.pubs)]

    with ReplayContext(cfg) as rc:
      pm = messaging.PubMaster(cfg.pubs)
      sockets = {s: messaging.sub_sock(s, timeout=100) for s in cfg.subs}

      managed_processes[cfg.proc_name].prepare()
      managed_processes[cfg.proc_name].start()

      if cfg.init_callback is not None:
        cfg.init_callback(rc, pm, all_msgs, fingerprint)
        CP = car.CarParams.from_bytes(Params().get("CarParams", block=True))

      log_msgs = []
      try:
        # Wait for process to startup
        with Timeout(10, error_msg=f"timed out waiting for process to start: {repr(cfg.proc_name)}"):
          while not all(pm.all_readers_updated(s) for s in cfg.pubs):
            time.sleep(0)

        for s in sockets.values():
          messaging.recv_one_or_none(s)

        # TODO wait for sockets to reconnect, for now lets just wait
        time.sleep(1)

        # Do the replay
        cnt = 0
        for msg in pub_msgs:
          with Timeout(cfg.timeout, error_msg=f"timed out testing process {repr(cfg.proc_name)}, {cnt}/{len(pub_msgs)} msgs done"):
            resp_sockets, should_recv = cfg.subs, True
            if cfg.should_recv_callback is not None:
              resp_sockets, should_recv = cfg.should_recv_callback(msg, CP, cfg, cnt)

            if len(log_msgs) == 0 and len(resp_sockets) > 0:
              for s in sockets.values():
                messaging.recv_one_or_none(s)
            
            rc.unlock_sockets(msg.which())
            pm.send(msg.which(), msg.as_builder())
            # wait for the next receive on the process side
            rc.wait_for_next_recv(msg.which(), should_recv)

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


def setup_env(CP=None, cfg=None, controlsState=None, lr=None, fingerprint=None):
  params = Params()
  params.clear_all()
  params.put_bool("OpenpilotEnabledToggle", True)
  params.put_bool("Passive", False)
  params.put_bool("DisengageOnAccelerator", True)
  params.put_bool("WideCameraOnly", False)
  params.put_bool("DisableLogging", False)

  os.environ["NO_RADAR_SLEEP"] = "1"
  os.environ["REPLAY"] = "1"
  if fingerprint is not None:
    os.environ['SKIP_FW_QUERY'] = "1"
    os.environ['FINGERPRINT'] = fingerprint
  else:
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

  if cfg.simulation:
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

    if fingerprint is None:
      if CP.fingerprintSource == "fw":
        params.put("CarParamsCache", CP.as_builder().to_bytes())
        os.environ['SKIP_FW_QUERY'] = ""
        os.environ['FINGERPRINT'] = ""
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
