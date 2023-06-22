#!/usr/bin/env python3
import os
import time
import signal
import platform
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Union, Any
from tqdm import tqdm
import capnp

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
from selfdrive.test.process_replay.migration import migrate_all
from tools.lib.logreader import LogReader

# Numpy gives different results based on CPU features after version 19
NUMPY_TOLERANCE = 1e-7
CI = "CI" in os.environ
TIMEOUT = 15
PROC_REPLAY_DIR = os.path.dirname(os.path.abspath(__file__))
FAKEDATA = os.path.join(PROC_REPLAY_DIR, "fakedata/")


class ReplayContext:
  def __init__(self, cfg):
    self.proc_name = cfg.proc_name
    self.pubs = cfg.pubs
    self.drained_pub = cfg.drained_pub
    assert(len(self.pubs) != 0 or self.drained_pub is not None)
  
  def __enter__(self):
    messaging.toggle_fake_events(True)
    messaging.set_fake_prefix(self.proc_name)

    if self.drained_pub is None:
      self.events = OrderedDict()
      for pub in self.pubs:
        self.events[pub] = messaging.fake_event_handle(pub, enable=True)
    else:
      self.events = {self.drained_pub: messaging.fake_event_handle(self.drained_pub, enable=True)}

    return self

  def __exit__(self, exc_type, exc_obj, exc_tb):
    del self.events

    messaging.toggle_fake_events(False)
    messaging.delete_fake_prefix()

  @property
  def all_recv_called_events(self):
    return [man.recv_called_event for man in self.events.values()]
  
  @property
  def all_recv_ready_events(self):
    return [man.recv_ready_event for man in self.events.values()]

  def send_sync(self, pm, endpoint, dat):
    self.events[endpoint].recv_called_event.wait()
    self.events[endpoint].recv_called_event.clear()
    pm.send(endpoint, dat)
    self.events[endpoint].recv_ready_event.set()

  def unlock_sockets(self):
    expected_sets = len(self.events)
    while expected_sets > 0:
      index = messaging.wait_for_one_event(self.all_recv_called_events)
      self.all_recv_called_events[index].clear()
      self.all_recv_ready_events[index].set()
      expected_sets -= 1

  def wait_for_recv_called(self):
    messaging.wait_for_one_event(self.all_recv_called_events)

  def wait_for_next_recv(self, trigger_empty_recv):
    index = messaging.wait_for_one_event(self.all_recv_called_events)
    if self.drained_pub is not None and trigger_empty_recv:
      self.all_recv_called_events[index].clear()
      self.all_recv_ready_events[index].set()
      self.all_recv_called_events[index].wait()


@dataclass
class ProcessConfig:
  proc_name: str
  pubs: List[str]
  subs: List[str]
  ignore: List[str]
  config_callback: Optional[Callable]
  init_callback: Optional[Callable]
  should_recv_callback: Optional[Callable]
  tolerance: Optional[float] = None
  environ: Dict[str, str] = field(default_factory=dict)
  subtest_name: str = ""
  field_tolerances: Dict[str, float] = field(default_factory=dict)
  timeout: int = 30
  simulation: bool = True
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
  rc.send_sync(pm, "can", messaging.new_message("can", 1))
  pm.send("pandaStates", messaging.new_message("pandaStates", 1))
  rc.send_sync(pm, "can", messaging.new_message("can", 1))
  rc.wait_for_next_recv(True)

  # fingerprinting is done, when CarParams is set
  while params.get("CarParams") is None:
    if len(canmsgs) == 0:
      raise ValueError("Fingerprinting failed. Run out of can msgs")

    m = canmsgs.pop(0)
    rc.send_sync(pm, "can", m.as_builder().to_bytes())
    rc.wait_for_next_recv(False)


def get_car_params_callback(rc, pm, msgs, fingerprint):
  if fingerprint:
    CarInterface, _, _ = interfaces[fingerprint]
    CP = CarInterface.get_non_essential_params(fingerprint)
  else:
    can = DummySocket()
    sendcan = DummySocket()

    canmsgs = [msg for msg in msgs if msg.which() == "can"]
    assert len(canmsgs) != 0, "CAN messages are required for carParams initialization"

    for m in canmsgs[:300]:
      can.send(m.as_builder().to_bytes())
    _, CP = get_car(can, sendcan, Params().get_bool("ExperimentalLongitudinalEnabled"))
  Params().put("CarParams", CP.to_bytes())


def controlsd_rcv_callback(msg, cfg, frame):
  # no sendcan until controlsd is initialized
  if msg.which() != "can":
    return False 

  socks = [
    s for s in cfg.subs if
    frame % int(service_list[msg.which()].frequency / service_list[s].frequency) == 0
  ]
  if "sendcan" in socks and (frame - 1) < 2000:
    socks.remove("sendcan")
  return len(socks) > 0


def radar_rcv_callback(msg, cfg, frame):
  return msg.which() == "can"


def calibration_rcv_callback(msg, cfg, frame):
  # calibrationd publishes 1 calibrationData every 5 cameraOdometry packets.
  # should_recv always true to increment frame
  return (frame - 1) == 0 or msg.which() == 'cameraOdometry'


def torqued_rcv_callback(msg, cfg, frame):
  # should_recv always true to increment frame
  return (frame - 1) == 0 or msg.which() == 'liveLocationKalman'


class FrequencyBasedRcvCallback:
  def __init__(self, trigger_msg_type):
    self.trigger_msg_type = trigger_msg_type

  def __call__(self, msg, cfg, frame):
    if msg.which() != self.trigger_msg_type:
      return False

    resp_sockets = [
      s for s in cfg.subs
      if frame % max(1, int(service_list[msg.which()].frequency / service_list[s].frequency)) == 0
    ]
    return bool(len(resp_sockets))


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
    drained_pub="can",
  ),
  ProcessConfig(
    proc_name="plannerd",
    pubs=["modelV2", "carControl", "carState", "controlsState", "radarState"],
    subs=["lateralPlan", "longitudinalPlan", "uiPlan"],
    ignore=["logMonoTime", "valid", "longitudinalPlan.processingDelay", "longitudinalPlan.solverExecutionTime", "lateralPlan.solverExecutionTime"],
    config_callback=None,
    init_callback=get_car_params_callback,
    should_recv_callback=FrequencyBasedRcvCallback("modelV2"),
    tolerance=NUMPY_TOLERANCE,
  ),
  ProcessConfig(
    proc_name="calibrationd",
    pubs=["carState", "cameraOdometry", "carParams"],
    subs=["liveCalibration"],
    ignore=["logMonoTime", "valid"],
    config_callback=None,
    init_callback=None,
    should_recv_callback=calibration_rcv_callback,
  ),
  ProcessConfig(
    proc_name="dmonitoringd",
    pubs=["driverStateV2", "liveCalibration", "carState", "modelV2", "controlsState"],
    subs=["driverMonitoringState"],
    ignore=["logMonoTime", "valid"],
    config_callback=None,
    init_callback=None,
    should_recv_callback=FrequencyBasedRcvCallback("driverStateV2"),
    tolerance=NUMPY_TOLERANCE,
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
    init_callback=None,
    should_recv_callback=None,
    tolerance=NUMPY_TOLERANCE,
  ),
  ProcessConfig(
    proc_name="paramsd",
    pubs=["liveLocationKalman", "carState"],
    subs=["liveParameters"],
    ignore=["logMonoTime", "valid"],
    config_callback=None,
    init_callback=get_car_params_callback,
    should_recv_callback=FrequencyBasedRcvCallback("liveLocationKalman"),
    tolerance=NUMPY_TOLERANCE,
  ),
  ProcessConfig(
    proc_name="ubloxd",
    pubs=["ubloxRaw"],
    subs=["ubloxGnss", "gpsLocationExternal"],
    ignore=["logMonoTime"],
    config_callback=None,
    init_callback=None,
    should_recv_callback=None,
  ),
  ProcessConfig(
    proc_name="laikad",
    pubs=["ubloxGnss", "qcomGnss"],
    subs=["gnssMeasurements"],
    ignore=["logMonoTime"],
    config_callback=laikad_config_pubsub_callback,
    init_callback=None,
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
  ),
]


def get_process_config(name):
  try:
    return next(c for c in CONFIGS if c.proc_name == name)
  except StopIteration as ex:
    raise Exception(f"Cannot find process config with name: {name}") from ex


def replay_process_with_name(name, lr, *args, **kwargs):
  cfg = get_process_config(name)
  return replay_process(cfg, lr, *args, **kwargs)


def replay_process(cfg, lr, fingerprint=None, return_all_logs=False, custom_params=None, disable_progress=False):
  all_msgs = migrate_all(lr, old_logtime=True)
  process_logs = _replay_single_process(cfg, all_msgs, fingerprint, custom_params or {}, disable_progress)

  if return_all_logs:
    keys = set(cfg.subs)
    modified_logs = [m for m in all_msgs if m.which() not in keys]
    modified_logs.extend(process_logs)
    modified_logs.sort(key=lambda m: m.logMonoTime)
    log_msgs = modified_logs
  else:
    log_msgs = process_logs

  return log_msgs


def _replay_single_process(
  cfg: ProcessConfig, lr: Union[LogReader, List[capnp._DynamicStructReader]], 
  fingerprint: Optional[str], custom_params: Optional[Dict[str, Any]], disable_progress: bool
):
  with OpenpilotPrefix():
    controlsState = None
    initialized = False
    if cfg.proc_name == "controlsd":
      for msg in lr:
        if msg.which() == "controlsState":
          controlsState = msg.controlsState
          if initialized:
            break
        elif msg.which() == "carEvents":
          initialized = car.CarEvent.EventName.controlsInitializing not in [e.name for e in msg.carEvents]

      assert controlsState is not None and initialized, "controlsState never initialized"

    if fingerprint is not None:
      setup_env(cfg=cfg, controlsState=controlsState, lr=lr, fingerprint=fingerprint, custom_params=custom_params)
    else:
      CP = next((m.carParams for m in lr if m.which() == "carParams"), None)
      assert CP is not None or "carParams" not in cfg.pubs, "carParams are missing and process needs it" 
      setup_env(CP=CP, cfg=cfg, controlsState=controlsState, lr=lr, custom_params=custom_params)

    if cfg.config_callback is not None:
      params = Params()
      cfg.pubs, cfg.drained_pub = cfg.config_callback(params, cfg)

    all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)
    pub_msgs = [msg for msg in all_msgs if msg.which() in set(cfg.pubs)]

    with ReplayContext(cfg) as rc:
      pm = messaging.PubMaster(cfg.pubs)
      sockets = {s: messaging.sub_sock(s, timeout=100) for s in cfg.subs}

      managed_processes[cfg.proc_name].prepare()
      managed_processes[cfg.proc_name].start()

      if cfg.init_callback is not None:
        cfg.init_callback(rc, pm, all_msgs, fingerprint)

      log_msgs, msg_queue = [], []
      try:
        # Wait for process to startup
        with Timeout(10, error_msg=f"timed out waiting for process to start: {repr(cfg.proc_name)}"):
          while not all(pm.all_readers_updated(s) for s in cfg.pubs):
            time.sleep(0)

        # Do the replay
        cnt = 0
        for msg in tqdm(pub_msgs, disable=disable_progress):
          with Timeout(cfg.timeout, error_msg=f"timed out testing process {repr(cfg.proc_name)}, {cnt}/{len(pub_msgs)} msgs done"):
            resp_sockets, end_of_cycle = cfg.subs, True
            if cfg.should_recv_callback is not None:
              end_of_cycle = cfg.should_recv_callback(msg, cfg, cnt)

            msg_queue.append(msg)
            if end_of_cycle:
              rc.wait_for_recv_called()

              # call recv to let sub-sockets reconnect, after we know the process is ready
              if cnt == 0:
                for s in sockets.values():
                  messaging.recv_one_or_none(s)

              # empty recv on drained pub indicates the end of messages, only do that if there're any
              trigger_empty_recv = False
              if cfg.drained_pub:
                trigger_empty_recv = next((True for m in msg_queue if m.which() == cfg.drained_pub), False)

              for m in msg_queue:
                pm.send(m.which(), m.as_builder())
              msg_queue = []

              rc.unlock_sockets()
              rc.wait_for_next_recv(trigger_empty_recv)

              for s in resp_sockets:
                ms = messaging.drain_sock(sockets[s])  
                for m in ms:
                  m = m.as_builder()
                  m.logMonoTime = msg.logMonoTime
                  log_msgs.append(m.as_reader())
              cnt += 1
          proc = managed_processes[cfg.proc_name].proc
          assert(proc and proc.is_alive())
      finally:
        managed_processes[cfg.proc_name].signal(signal.SIGKILL)
        managed_processes[cfg.proc_name].stop()

      return log_msgs


def setup_env(cfg=None, CP=None, controlsState=None, lr=None, fingerprint=None, custom_params=None, log_dir=None):
  if platform.system() != "Darwin":
    os.environ["PARAMS_ROOT"] = "/dev/shm/params"
  if log_dir is not None:
    os.environ["LOG_ROOT"] = log_dir

  params = Params()
  params.clear_all()
  params.put_bool("OpenpilotEnabledToggle", True)
  params.put_bool("Passive", False)
  params.put_bool("DisengageOnAccelerator", True)
  params.put_bool("DisableLogging", False)
  if custom_params is not None:
    for k, v in custom_params.items():
      if type(v) == bool:
        params.put_bool(k, v)
      else:
        params.put(k, v)

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

  if cfg is not None and cfg.simulation:
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


def check_openpilot_enabled(msgs):
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
