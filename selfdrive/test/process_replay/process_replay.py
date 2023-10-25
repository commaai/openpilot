#!/usr/bin/env python3
import os
import time
import copy
import json
import heapq
import signal
import platform
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Union, Any, Iterable, Tuple
from tqdm import tqdm
import capnp

import cereal.messaging as messaging
from cereal import car
from cereal.services import SERVICE_LIST
from cereal.visionipc import VisionIpcServer, get_endpoint_name as vipc_get_endpoint_name
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.common.timeout import Timeout
from openpilot.common.realtime import DT_CTRL
from panda.python import ALTERNATIVE_EXPERIENCE
from openpilot.selfdrive.car.car_helpers import get_car, interfaces
from openpilot.selfdrive.manager.process_config import managed_processes
from openpilot.selfdrive.test.process_replay.vision_meta import meta_from_camera_state, available_streams
from openpilot.selfdrive.test.process_replay.migration import migrate_all
from openpilot.selfdrive.test.process_replay.capture import ProcessOutputCapture
from openpilot.tools.lib.logreader import LogIterable

# Numpy gives different results based on CPU features after version 19
NUMPY_TOLERANCE = 1e-7
PROC_REPLAY_DIR = os.path.dirname(os.path.abspath(__file__))
FAKEDATA = os.path.join(PROC_REPLAY_DIR, "fakedata/")

class DummySocket:
  def __init__(self):
    self.data: List[bytes] = []

  def receive(self, non_blocking: bool = False) -> Optional[bytes]:
    if non_blocking:
      return None

    return self.data.pop()

  def send(self, data: bytes):
    self.data.append(data)

class LauncherWithCapture:
  def __init__(self, capture: ProcessOutputCapture, launcher: Callable):
    self.capture = capture
    self.launcher = launcher

  def __call__(self, *args, **kwargs):
    self.capture.link_with_current_proc()
    self.launcher(*args, **kwargs)


class ReplayContext:
  def __init__(self, cfg):
    self.proc_name = cfg.proc_name
    self.pubs = cfg.pubs
    self.main_pub = cfg.main_pub
    self.main_pub_drained = cfg.main_pub_drained
    self.unlocked_pubs = cfg.unlocked_pubs
    assert(len(self.pubs) != 0 or self.main_pub is not None)

  def __enter__(self):
    self.open_context()

    return self

  def __exit__(self, exc_type, exc_obj, exc_tb):
    self.close_context()

  def open_context(self):
    messaging.toggle_fake_events(True)
    messaging.set_fake_prefix(self.proc_name)

    if self.main_pub is None:
      self.events = OrderedDict()
      pubs_with_events = [pub for pub in self.pubs if pub not in self.unlocked_pubs]
      for pub in pubs_with_events:
        self.events[pub] = messaging.fake_event_handle(pub, enable=True)
    else:
      self.events = {self.main_pub: messaging.fake_event_handle(self.main_pub, enable=True)}

  def close_context(self):
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
    if self.main_pub is not None and self.main_pub_drained and trigger_empty_recv:
      self.all_recv_called_events[index].clear()
      self.all_recv_ready_events[index].set()
      self.all_recv_called_events[index].wait()


@dataclass
class ProcessConfig:
  proc_name: str
  pubs: List[str]
  subs: List[str]
  ignore: List[str]
  config_callback: Optional[Callable] = None
  init_callback: Optional[Callable] = None
  should_recv_callback: Optional[Callable] = None
  tolerance: Optional[float] = None
  processing_time: float = 0.001
  timeout: int = 30
  simulation: bool = True
  main_pub: Optional[str] = None
  main_pub_drained: bool = True
  vision_pubs: List[str] = field(default_factory=list)
  ignore_alive_pubs: List[str] = field(default_factory=list)
  unlocked_pubs: List[str] = field(default_factory=list)


class ProcessContainer:
  def __init__(self, cfg: ProcessConfig):
    self.prefix = OpenpilotPrefix(clean_dirs_on_exit=False)
    self.cfg = copy.deepcopy(cfg)
    self.process = copy.deepcopy(managed_processes[cfg.proc_name])
    self.msg_queue: List[capnp._DynamicStructReader] = []
    self.cnt = 0
    self.pm: Optional[messaging.PubMaster] = None
    self.sockets: Optional[List[messaging.SubSocket]] = None
    self.rc: Optional[ReplayContext] = None
    self.vipc_server: Optional[VisionIpcServer] = None
    self.environ_config: Optional[Dict[str, Any]] = None
    self.capture: Optional[ProcessOutputCapture] = None

  @property
  def has_empty_queue(self) -> bool:
    return len(self.msg_queue) == 0

  @property
  def pubs(self) -> List[str]:
    return self.cfg.pubs

  @property
  def subs(self) -> List[str]:
    return self.cfg.subs

  def _clean_env(self):
    for k in self.environ_config.keys():
      if k in os.environ:
        del os.environ[k]

    for k in ["PROC_NAME", "SIMULATION"]:
      if k in os.environ:
        del os.environ[k]

  def _setup_env(self, params_config: Dict[str, Any], environ_config: Dict[str, Any]):
    for k, v in environ_config.items():
      if len(v) != 0:
        os.environ[k] = v
      elif k in os.environ:
        del os.environ[k]

    os.environ["PROC_NAME"] = self.cfg.proc_name
    if self.cfg.simulation:
      os.environ["SIMULATION"] = "1"
    elif "SIMULATION" in os.environ:
      del os.environ["SIMULATION"]

    params = Params()
    for k, v in params_config.items():
      if isinstance(v, bool):
        params.put_bool(k, v)
      else:
        params.put(k, v)

    self.environ_config = environ_config

  def _setup_vision_ipc(self, all_msgs):
    assert len(self.cfg.vision_pubs) != 0

    device_type = next(str(msg.initData.deviceType) for msg in all_msgs if msg.which() == "initData")

    vipc_server = VisionIpcServer("camerad")
    streams_metas = available_streams(all_msgs)
    for meta in streams_metas:
      if meta.camera_state in self.cfg.vision_pubs:
        vipc_server.create_buffers(meta.stream, 2, False, *meta.frame_sizes[device_type])
    vipc_server.start_listener()

    self.vipc_server = vipc_server
    self.cfg.vision_pubs = [meta.camera_state for meta in streams_metas if meta.camera_state in self.cfg.vision_pubs]

  def _start_process(self):
    if self.capture is not None:
      self.process.launcher = LauncherWithCapture(self.capture, self.process.launcher)
    self.process.prepare()
    self.process.start()

  def start(
    self, params_config: Dict[str, Any], environ_config: Dict[str, Any],
    all_msgs: LogIterable,
    fingerprint: Optional[str], capture_output: bool
  ):
    with self.prefix as p:
      self._setup_env(params_config, environ_config)

      if self.cfg.config_callback is not None:
        params = Params()
        self.cfg.config_callback(params, self.cfg, all_msgs)

      self.rc = ReplayContext(self.cfg)
      self.rc.open_context()

      self.pm = messaging.PubMaster(self.cfg.pubs)
      self.sockets = [messaging.sub_sock(s, timeout=100) for s in self.cfg.subs]

      if len(self.cfg.vision_pubs) != 0:
        self._setup_vision_ipc(all_msgs)
        assert self.vipc_server is not None

      if capture_output:
        self.capture = ProcessOutputCapture(self.cfg.proc_name, p.prefix)

      self._start_process()

      if self.cfg.init_callback is not None:
        self.cfg.init_callback(self.rc, self.pm, all_msgs, fingerprint)

      # wait for process to startup
      with Timeout(10, error_msg=f"timed out waiting for process to start: {repr(self.cfg.proc_name)}"):
        while not all(self.pm.all_readers_updated(s) for s in self.cfg.pubs if s not in self.cfg.ignore_alive_pubs):
          time.sleep(0)

  def stop(self):
    with self.prefix:
      self.process.signal(signal.SIGKILL)
      self.process.stop()
      self.rc.close_context()
      self.prefix.clean_dirs()
      self._clean_env()

  def run_step(self, msg: capnp._DynamicStructReader, frs: Optional[Dict[str, Any]]) -> List[capnp._DynamicStructReader]:
    assert self.rc and self.pm and self.sockets and self.process.proc

    output_msgs = []
    with self.prefix, Timeout(self.cfg.timeout, error_msg=f"timed out testing process {repr(self.cfg.proc_name)}"):
      end_of_cycle = True
      if self.cfg.should_recv_callback is not None:
        end_of_cycle = self.cfg.should_recv_callback(msg, self.cfg, self.cnt)

      self.msg_queue.append(msg)
      if end_of_cycle:
        self.rc.wait_for_recv_called()

        # call recv to let sub-sockets reconnect, after we know the process is ready
        if self.cnt == 0:
          for s in self.sockets:
            messaging.recv_one_or_none(s)

        # empty recv on drained pub indicates the end of messages, only do that if there're any
        trigger_empty_recv = False
        if self.cfg.main_pub and self.cfg.main_pub_drained:
          trigger_empty_recv = next((True for m in self.msg_queue if m.which() == self.cfg.main_pub), False)

        for m in self.msg_queue:
          self.pm.send(m.which(), m.as_builder())
          # send frames if needed
          if self.vipc_server is not None and m.which() in self.cfg.vision_pubs:
            camera_state = getattr(m, m.which())
            camera_meta = meta_from_camera_state(m.which())
            assert frs is not None
            img = frs[m.which()].get(camera_state.frameId, pix_fmt="nv12")[0]
            self.vipc_server.send(camera_meta.stream, img.flatten().tobytes(),
                                  camera_state.frameId, camera_state.timestampSof, camera_state.timestampEof)
        self.msg_queue = []

        self.rc.unlock_sockets()
        self.rc.wait_for_next_recv(trigger_empty_recv)

        for socket in self.sockets:
          ms = messaging.drain_sock(socket)
          for m in ms:
            m = m.as_builder()
            m.logMonoTime = msg.logMonoTime + int(self.cfg.processing_time * 1e9)
            output_msgs.append(m.as_reader())
        self.cnt += 1
    assert self.process.proc.is_alive()

    return output_msgs


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
  params = Params()
  if fingerprint:
    CarInterface, _, _ = interfaces[fingerprint]
    CP = CarInterface.get_non_essential_params(fingerprint)
  else:
    can = DummySocket()
    sendcan = DummySocket()

    canmsgs = [msg for msg in msgs if msg.which() == "can"]
    has_cached_cp = params.get("CarParamsCache") is not None
    assert len(canmsgs) != 0, "CAN messages are required for fingerprinting"
    assert os.environ.get("SKIP_FW_QUERY", False) or has_cached_cp, \
            "CarParamsCache is required for fingerprinting. Make sure to keep carParams msgs in the logs."

    for m in canmsgs[:300]:
      can.send(m.as_builder().to_bytes())
    _, CP = get_car(can, sendcan, Params().get_bool("ExperimentalLongitudinalEnabled"))
  params.put("CarParams", CP.to_bytes())
  return CP


def controlsd_rcv_callback(msg, cfg, frame):
  # no sendcan until controlsd is initialized
  if msg.which() != "can":
    return False

  socks = [
    s for s in cfg.subs if
    frame % int(SERVICE_LIST[msg.which()].frequency / SERVICE_LIST[s].frequency) == 0
  ]
  if "sendcan" in socks and (frame - 1) < 2000:
    socks.remove("sendcan")
  return len(socks) > 0


def calibration_rcv_callback(msg, cfg, frame):
  # calibrationd publishes 1 calibrationData every 5 cameraOdometry packets.
  # should_recv always true to increment frame
  return (frame - 1) == 0 or msg.which() == 'cameraOdometry'


def torqued_rcv_callback(msg, cfg, frame):
  # should_recv always true to increment frame
  return (frame - 1) == 0 or msg.which() == 'liveLocationKalman'


def dmonitoringmodeld_rcv_callback(msg, cfg, frame):
  return msg.which() == "driverCameraState"


class ModeldCameraSyncRcvCallback:
  def __init__(self):
    self.road_present = False
    self.wide_road_present = False
    self.is_dual_camera = True

  def __call__(self, msg, cfg, frame):
    self.is_dual_camera = len(cfg.vision_pubs) == 2
    if msg.which() == "roadCameraState":
      self.road_present = True
    elif msg.which() == "wideRoadCameraState":
      self.wide_road_present = True

    if self.road_present and self.wide_road_present:
      self.road_present, self.wide_road_present = False, False
      return True
    elif self.road_present and not self.is_dual_camera:
      self.road_present = False
      return True
    else:
      return False


class MessageBasedRcvCallback:
  def __init__(self, trigger_msg_type):
    self.trigger_msg_type = trigger_msg_type

  def __call__(self, msg, cfg, frame):
    return msg.which() == self.trigger_msg_type


class FrequencyBasedRcvCallback:
  def __init__(self, trigger_msg_type):
    self.trigger_msg_type = trigger_msg_type

  def __call__(self, msg, cfg, frame):
    if msg.which() != self.trigger_msg_type:
      return False

    resp_sockets = [
      s for s in cfg.subs
      if frame % max(1, int(SERVICE_LIST[msg.which()].frequency / SERVICE_LIST[s].frequency)) == 0
    ]
    return bool(len(resp_sockets))


def controlsd_config_callback(params, cfg, lr):
  controlsState = None
  initialized = False
  for msg in lr:
    if msg.which() == "controlsState":
      controlsState = msg.controlsState
      if initialized:
        break
    elif msg.which() == "carEvents":
      initialized = car.CarEvent.EventName.controlsInitializing not in [e.name for e in msg.carEvents]

  assert controlsState is not None and initialized, "controlsState never initialized"
  params.put("ReplayControlsState", controlsState.as_builder().to_bytes())


def locationd_config_pubsub_callback(params, cfg, lr):
  ublox = params.get_bool("UbloxAvailable")
  sub_keys = ({"gpsLocation", } if ublox else {"gpsLocationExternal", })

  cfg.pubs = set(cfg.pubs) - sub_keys


CONFIGS = [
  ProcessConfig(
    proc_name="controlsd",
    pubs=[
      "can", "deviceState", "pandaStates", "peripheralState", "liveCalibration", "driverMonitoringState",
      "longitudinalPlan", "lateralPlan", "liveLocationKalman", "liveParameters", "radarState",
      "modelV2", "driverCameraState", "roadCameraState", "wideRoadCameraState", "managerState",
      "testJoystick", "liveTorqueParameters", "accelerometer", "gyroscope"
    ],
    subs=["controlsState", "carState", "carControl", "sendcan", "carEvents", "carParams"],
    ignore=["logMonoTime", "valid", "controlsState.startMonoTime", "controlsState.cumLagMs"],
    config_callback=controlsd_config_callback,
    init_callback=controlsd_fingerprint_callback,
    should_recv_callback=controlsd_rcv_callback,
    tolerance=NUMPY_TOLERANCE,
    processing_time=0.004,
    main_pub="can",
  ),
  ProcessConfig(
    proc_name="radard",
    pubs=["can", "carState", "modelV2"],
    subs=["radarState", "liveTracks"],
    ignore=["logMonoTime", "valid", "radarState.cumLagMs"],
    init_callback=get_car_params_callback,
    should_recv_callback=MessageBasedRcvCallback("can"),
    main_pub="can",
  ),
  ProcessConfig(
    proc_name="plannerd",
    pubs=["modelV2", "carControl", "carState", "controlsState", "radarState"],
    subs=["lateralPlan", "longitudinalPlan", "uiPlan"],
    ignore=["logMonoTime", "valid", "longitudinalPlan.processingDelay", "longitudinalPlan.solverExecutionTime", "lateralPlan.solverExecutionTime"],
    init_callback=get_car_params_callback,
    should_recv_callback=FrequencyBasedRcvCallback("modelV2"),
    tolerance=NUMPY_TOLERANCE,
  ),
  ProcessConfig(
    proc_name="calibrationd",
    pubs=["carState", "cameraOdometry", "carParams"],
    subs=["liveCalibration"],
    ignore=["logMonoTime", "valid"],
    should_recv_callback=calibration_rcv_callback,
  ),
  ProcessConfig(
    proc_name="dmonitoringd",
    pubs=["driverStateV2", "liveCalibration", "carState", "modelV2", "controlsState"],
    subs=["driverMonitoringState"],
    ignore=["logMonoTime", "valid"],
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
    tolerance=NUMPY_TOLERANCE,
  ),
  ProcessConfig(
    proc_name="paramsd",
    pubs=["liveLocationKalman", "carState"],
    subs=["liveParameters"],
    ignore=["logMonoTime", "valid"],
    init_callback=get_car_params_callback,
    should_recv_callback=FrequencyBasedRcvCallback("liveLocationKalman"),
    tolerance=NUMPY_TOLERANCE,
    processing_time=0.004,
  ),
  ProcessConfig(
    proc_name="ubloxd",
    pubs=["ubloxRaw"],
    subs=["ubloxGnss", "gpsLocationExternal"],
    ignore=["logMonoTime"],
  ),
  ProcessConfig(
    proc_name="torqued",
    pubs=["liveLocationKalman", "carState", "carControl"],
    subs=["liveTorqueParameters"],
    ignore=["logMonoTime"],
    init_callback=get_car_params_callback,
    should_recv_callback=torqued_rcv_callback,
    tolerance=NUMPY_TOLERANCE,
  ),
  ProcessConfig(
    proc_name="modeld",
    pubs=["lateralPlan", "roadCameraState", "wideRoadCameraState", "liveCalibration", "driverMonitoringState"],
    subs=["modelV2", "cameraOdometry"],
    ignore=["logMonoTime", "modelV2.frameDropPerc", "modelV2.modelExecutionTime"],
    should_recv_callback=ModeldCameraSyncRcvCallback(),
    tolerance=NUMPY_TOLERANCE,
    processing_time=0.020,
    main_pub=vipc_get_endpoint_name("camerad", meta_from_camera_state("roadCameraState").stream),
    main_pub_drained=False,
    vision_pubs=["roadCameraState", "wideRoadCameraState"],
    ignore_alive_pubs=["wideRoadCameraState"],
  ),
  ProcessConfig(
    proc_name="dmonitoringmodeld",
    pubs=["liveCalibration", "driverCameraState"],
    subs=["driverStateV2"],
    ignore=["logMonoTime", "driverStateV2.modelExecutionTime", "driverStateV2.dspExecutionTime"],
    should_recv_callback=dmonitoringmodeld_rcv_callback,
    tolerance=NUMPY_TOLERANCE,
    processing_time=0.020,
    main_pub=vipc_get_endpoint_name("camerad", meta_from_camera_state("driverCameraState").stream),
    main_pub_drained=False,
    vision_pubs=["driverCameraState"],
    ignore_alive_pubs=["driverCameraState"],
  ),
]


def get_process_config(name: str) -> ProcessConfig:
  try:
    return copy.deepcopy(next(c for c in CONFIGS if c.proc_name == name))
  except StopIteration as ex:
    raise Exception(f"Cannot find process config with name: {name}") from ex


def get_custom_params_from_lr(lr: LogIterable, initial_state: str = "first") -> Dict[str, Any]:
  """
  Use this to get custom params dict based on provided logs.
  Useful when replaying following processes: calibrationd, paramsd, torqued
  The params may be based on first or last message of given type (carParams, liveCalibration, liveParameters, liveTorqueParameters) in the logs.
  """

  car_params = [m for m in lr if m.which() == "carParams"]
  live_calibration = [m for m in lr if m.which() == "liveCalibration"]
  live_parameters = [m for m in lr if m.which() == "liveParameters"]
  live_torque_parameters = [m for m in lr if m.which() == "liveTorqueParameters"]

  assert initial_state in ["first", "last"]
  msg_index = 0 if initial_state == "first" else -1

  assert len(car_params) > 0, "carParams required for initial state of liveParameters and liveTorqueCarParams"
  CP = car_params[msg_index].carParams

  custom_params = {}
  if len(live_calibration) > 0:
    custom_params["CalibrationParams"] = live_calibration[msg_index].as_builder().to_bytes()
  if len(live_parameters) > 0:
    lp_dict = live_parameters[msg_index].to_dict()
    lp_dict["carFingerprint"] = CP.carFingerprint
    custom_params["LiveParameters"] = json.dumps(lp_dict)
  if len(live_torque_parameters) > 0:
    custom_params["LiveTorqueCarParams"] = CP.as_builder().to_bytes()
    custom_params["LiveTorqueParameters"] = live_torque_parameters[msg_index].as_builder().to_bytes()

  return custom_params


def replay_process_with_name(name: Union[str, Iterable[str]], lr: LogIterable, *args, **kwargs) -> List[capnp._DynamicStructReader]:
  if isinstance(name, str):
    cfgs = [get_process_config(name)]
  elif isinstance(name, Iterable):
    cfgs = [get_process_config(n) for n in name]
  else:
    raise ValueError("name must be str or collections of strings")

  return replay_process(cfgs, lr, *args, **kwargs)


def replay_process(
  cfg: Union[ProcessConfig, Iterable[ProcessConfig]], lr: LogIterable, frs: Optional[Dict[str, Any]] = None,
  fingerprint: Optional[str] = None, return_all_logs: bool = False, custom_params: Optional[Dict[str, Any]] = None,
  captured_output_store: Optional[Dict[str, Dict[str, str]]] = None, disable_progress: bool = False
) -> List[capnp._DynamicStructReader]:
  if isinstance(cfg, Iterable):
    cfgs = list(cfg)
  else:
    cfgs = [cfg]

  all_msgs = migrate_all(lr, old_logtime=True,
                         manager_states=True,
                         panda_states=any("pandaStates" in cfg.pubs for cfg in cfgs),
                         camera_states=any(len(cfg.vision_pubs) != 0 for cfg in cfgs))
  process_logs = _replay_multi_process(cfgs, all_msgs, frs, fingerprint, custom_params, captured_output_store, disable_progress)

  if return_all_logs:
    keys = {m.which() for m in process_logs}
    modified_logs = [m for m in all_msgs if m.which() not in keys]
    modified_logs.extend(process_logs)
    modified_logs.sort(key=lambda m: int(m.logMonoTime))
    log_msgs = modified_logs
  else:
    log_msgs = process_logs

  return log_msgs


def _replay_multi_process(
  cfgs: List[ProcessConfig], lr: LogIterable, frs: Optional[Dict[str, Any]], fingerprint: Optional[str],
  custom_params: Optional[Dict[str, Any]], captured_output_store: Optional[Dict[str, Dict[str, str]]], disable_progress: bool
) -> List[capnp._DynamicStructReader]:
  if fingerprint is not None:
    params_config = generate_params_config(lr=lr, fingerprint=fingerprint, custom_params=custom_params)
    env_config = generate_environ_config(fingerprint=fingerprint)
  else:
    CP = next((m.carParams for m in lr if m.which() == "carParams"), None)
    params_config = generate_params_config(lr=lr, CP=CP, custom_params=custom_params)
    env_config = generate_environ_config(CP=CP)

  # validate frs and vision pubs
  all_vision_pubs = [pub for cfg in cfgs for pub in cfg.vision_pubs]
  if len(all_vision_pubs) != 0:
    assert frs is not None, "frs must be provided when replaying process using vision streams"
    assert all(meta_from_camera_state(st) is not None for st in all_vision_pubs), \
                                                          f"undefined vision stream spotted, probably misconfigured process: (vision pubs: {all_vision_pubs})"
    required_vision_pubs = {m.camera_state for m in available_streams(lr)} & set(all_vision_pubs)
    assert all(st in frs for st in required_vision_pubs), f"frs for this process must contain following vision streams: {required_vision_pubs}"

  all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)
  log_msgs = []
  try:
    containers = []
    for cfg in cfgs:
      container = ProcessContainer(cfg)
      containers.append(container)
      container.start(params_config, env_config, all_msgs, fingerprint, captured_output_store is not None)

    all_pubs = {pub for container in containers for pub in container.pubs}
    all_subs = {sub for container in containers for sub in container.subs}
    lr_pubs = all_pubs - all_subs
    pubs_to_containers = {pub: [container for container in containers if pub in container.pubs] for pub in all_pubs}

    pub_msgs = [msg for msg in all_msgs if msg.which() in lr_pubs]
    # external queue for messages taken from logs; internal queue for messages generated by processes, which will be republished
    external_pub_queue: List[capnp._DynamicStructReader] = pub_msgs.copy()
    internal_pub_queue: List[capnp._DynamicStructReader] = []
    # heap for maintaining the order of messages generated by processes, where each element: (logMonoTime, index in internal_pub_queue)
    internal_pub_index_heap: List[Tuple[int, int]] = []

    pbar = tqdm(total=len(external_pub_queue), disable=disable_progress)
    while len(external_pub_queue) != 0 or (len(internal_pub_index_heap) != 0 and not all(c.has_empty_queue for c in containers)):
      if len(internal_pub_index_heap) == 0 or (len(external_pub_queue) != 0 and external_pub_queue[0].logMonoTime < internal_pub_index_heap[0][0]):
        msg = external_pub_queue.pop(0)
        pbar.update(1)
      else:
        _, index = heapq.heappop(internal_pub_index_heap)
        msg = internal_pub_queue[index]

      target_containers = pubs_to_containers[msg.which()]
      for container in target_containers:
        output_msgs = container.run_step(msg, frs)
        for m in output_msgs:
          if m.which() in all_pubs:
            internal_pub_queue.append(m)
            heapq.heappush(internal_pub_index_heap, (m.logMonoTime, len(internal_pub_queue) - 1))
        log_msgs.extend(output_msgs)
  finally:
    for container in containers:
      container.stop()
      if captured_output_store is not None:
        assert container.capture is not None
        out, err = container.capture.read_outerr()
        captured_output_store[container.cfg.proc_name] = {"out": out, "err": err}

  return log_msgs


def generate_params_config(lr=None, CP=None, fingerprint=None, custom_params=None) -> Dict[str, Any]:
  params_dict = {
    "OpenpilotEnabledToggle": True,
    "Passive": False,
    "DisengageOnAccelerator": True,
    "DisableLogging": False,
  }

  if custom_params is not None:
    params_dict.update(custom_params)
  if lr is not None:
    has_ublox = any(msg.which() == "ubloxGnss" for msg in lr)
    params_dict["UbloxAvailable"] = has_ublox
    is_rhd = next((msg.driverMonitoringState.isRHD for msg in lr if msg.which() == "driverMonitoringState"), False)
    params_dict["IsRhdDetected"] = is_rhd

  if CP is not None:
    if CP.alternativeExperience == ALTERNATIVE_EXPERIENCE.DISABLE_DISENGAGE_ON_GAS:
      params_dict["DisengageOnAccelerator"] = False

    if fingerprint is None:
      if CP.fingerprintSource == "fw":
        params_dict["CarParamsCache"] = CP.as_builder().to_bytes()

    if CP.openpilotLongitudinalControl:
      params_dict["ExperimentalLongitudinalEnabled"] = True

  return params_dict


def generate_environ_config(CP=None, fingerprint=None, log_dir=None) -> Dict[str, Any]:
  environ_dict = {}
  if platform.system() != "Darwin":
    environ_dict["PARAMS_ROOT"] = "/dev/shm/params"
  if log_dir is not None:
    environ_dict["LOG_ROOT"] = log_dir

  environ_dict["NO_RADAR_SLEEP"] = "1"
  environ_dict["REPLAY"] = "1"

  # Regen or python process
  if CP is not None and fingerprint is None:
    if CP.fingerprintSource == "fw":
      environ_dict['SKIP_FW_QUERY'] = ""
      environ_dict['FINGERPRINT'] = ""
    else:
      environ_dict['SKIP_FW_QUERY'] = "1"
      environ_dict['FINGERPRINT'] = CP.carFingerprint
  elif fingerprint is not None:
    environ_dict['SKIP_FW_QUERY'] = "1"
    environ_dict['FINGERPRINT'] = fingerprint
  else:
    environ_dict["SKIP_FW_QUERY"] = ""
    environ_dict["FINGERPRINT"] = ""

  return environ_dict


def check_openpilot_enabled(msgs: LogIterable) -> bool:
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
