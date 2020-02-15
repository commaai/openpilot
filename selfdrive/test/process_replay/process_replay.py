#!/usr/bin/env python3
import os
import sys
import threading
import importlib

if "CI" in os.environ:
  tqdm = lambda x: x
else:
  from tqdm import tqdm

from cereal import car, log
from selfdrive.car.car_helpers import get_car
import selfdrive.manager as manager
import cereal.messaging as messaging
from common.params import Params
from cereal.services import service_list
from collections import namedtuple

ProcessConfig = namedtuple('ProcessConfig', ['proc_name', 'pub_sub', 'ignore', 'init_callback', 'should_recv_callback'])

def wait_for_event(evt):
  if not evt.wait(15):
    if threading.currentThread().getName() == "MainThread":
      # tested process likely died. don't let test just hang
      raise Exception("Timeout reached. Tested process likely crashed.")
    else:
      # done testing this process, let it die
      sys.exit(0)

class FakeSocket:
  def __init__(self, wait=True):
    self.data = []
    self.wait = wait
    self.recv_called = threading.Event()
    self.recv_ready = threading.Event()

  def receive(self, non_blocking=False):
    if non_blocking:
      return None

    if self.wait:
      self.recv_called.set()
      wait_for_event(self.recv_ready)
      self.recv_ready.clear()
    return self.data.pop()

  def send(self, data):
    if self.wait:
      wait_for_event(self.recv_called)
      self.recv_called.clear()

    self.data.append(data)

    if self.wait:
      self.recv_ready.set()

  def wait_for_recv(self):
    wait_for_event(self.recv_called)

class DumbSocket:
  def __init__(self, s=None):
    if s is not None:
      dat = messaging.new_message()
      dat.init(s)
      self.data = dat.to_bytes()

  def receive(self, non_blocking=False):
    return self.data

  def send(self, dat):
    pass

class FakeSubMaster(messaging.SubMaster):
  def __init__(self, services):
    super(FakeSubMaster, self).__init__(services, addr=None)
    self.sock = {s: DumbSocket(s) for s in services}
    self.update_called = threading.Event()
    self.update_ready = threading.Event()

    self.wait_on_getitem = False

  def __getitem__(self, s):
    # hack to know when fingerprinting is done
    if self.wait_on_getitem:
      self.update_called.set()
      wait_for_event(self.update_ready)
      self.update_ready.clear()
    return self.data[s]

  def update(self, timeout=-1):
    self.update_called.set()
    wait_for_event(self.update_ready)
    self.update_ready.clear()

  def update_msgs(self, cur_time, msgs):
    wait_for_event(self.update_called)
    self.update_called.clear()
    super(FakeSubMaster, self).update_msgs(cur_time, msgs)
    self.update_ready.set()

  def wait_for_update(self):
    wait_for_event(self.update_called)

class FakePubMaster(messaging.PubMaster):
  def __init__(self, services):
    self.data = {}
    self.sock = {}
    self.last_updated = None
    for s in services:
      data = messaging.new_message()
      try:
        data.init(s)
      except:
        data.init(s, 0)
      self.data[s] = data.as_reader()
      self.sock[s] = DumbSocket()
    self.send_called = threading.Event()
    self.get_called = threading.Event()

  def send(self, s, dat):
    self.last_updated = s
    if isinstance(dat, bytes):
      self.data[s] = log.Event.from_bytes(dat)
    else:
      self.data[s] = dat.as_reader()
    self.send_called.set()
    wait_for_event(self.get_called)
    self.get_called.clear()

  def wait_for_msg(self):
    wait_for_event(self.send_called)
    self.send_called.clear()
    dat = self.data[self.last_updated]
    self.get_called.set()
    return dat

def fingerprint(msgs, fsm, can_sock):
  print("start fingerprinting")
  fsm.wait_on_getitem = True

  # populate fake socket with data for fingerprinting
  canmsgs = [msg for msg in msgs if msg.which() == "can"]
  wait_for_event(can_sock.recv_called)
  can_sock.recv_called.clear()
  can_sock.data = [msg.as_builder().to_bytes() for msg in canmsgs[:300]]
  can_sock.recv_ready.set()
  can_sock.wait = False

  # we know fingerprinting is done when controlsd sets sm['pathPlan'].sensorValid
  wait_for_event(fsm.update_called)
  fsm.update_called.clear()

  fsm.wait_on_getitem = False
  can_sock.wait = True
  can_sock.data = []

  fsm.update_ready.set()
  print("finished fingerprinting")

def get_car_params(msgs, fsm, can_sock):
  can = FakeSocket(wait=False)
  sendcan = FakeSocket(wait=False)

  canmsgs = [msg for msg in msgs if msg.which() == 'can']
  for m in canmsgs[:300]:
    can.send(m.as_builder().to_bytes())
  _, CP = get_car(can, sendcan)
  Params().put("CarParams", CP.to_bytes())

def radar_rcv_callback(msg, CP, cfg, fsm):
  if msg.which() != "can":
    return [], False
  elif CP.radarOffCan:
    return ["radarState", "liveTracks"], True

  radar_msgs = {"honda": [0x445], "toyota": [0x19f, 0x22f], "gm": [0x474],
                "chrysler": [0x2d4]}.get(CP.carName, None)

  if radar_msgs is None:
    raise NotImplementedError

  for m in msg.can:
    if m.src == 1 and m.address in radar_msgs:
      return ["radarState", "liveTracks"], True
  return [], False

def calibration_rcv_callback(msg, CP, cfg, fsm):
  # calibrationd publishes 1 calibrationData every 5 cameraOdometry packets.
  # should_recv always true to increment frame
  recv_socks = ["liveCalibration"] if (fsm.frame + 1) % 5 == 0 else []
  return recv_socks, True


CONFIGS = [
  ProcessConfig(
    proc_name="controlsd",
    pub_sub={
      "can": ["controlsState", "carState", "carControl", "sendcan", "carEvents", "carParams"],
      "thermal": [], "health": [], "liveCalibration": [], "dMonitoringState": [], "plan": [], "pathPlan": [], "gpsLocation": [],
      "model": [],
    },
    ignore=[("logMonoTime", 0), ("valid", True), ("controlsState.startMonoTime", 0), ("controlsState.cumLagMs", 0)],
    init_callback=fingerprint,
    should_recv_callback=None,
  ),
  ProcessConfig(
    proc_name="radard",
    pub_sub={
      "can": ["radarState", "liveTracks"],
      "liveParameters":  [], "controlsState":  [], "model":  [],
    },
    ignore=[("logMonoTime", 0), ("valid", True), ("radarState.cumLagMs", 0)],
    init_callback=get_car_params,
    should_recv_callback=radar_rcv_callback,
  ),
  ProcessConfig(
    proc_name="plannerd",
    pub_sub={
      "model": ["pathPlan"], "radarState": ["plan"],
      "carState": [], "controlsState": [], "liveParameters": [],
    },
    ignore=[("logMonoTime", 0), ("valid", True), ("plan.processingDelay", 0)],
    init_callback=get_car_params,
    should_recv_callback=None,
  ),
  ProcessConfig(
    proc_name="calibrationd",
    pub_sub={
      "cameraOdometry": ["liveCalibration"]
    },
    ignore=[("logMonoTime", 0), ("valid", True)],
    init_callback=get_car_params,
    should_recv_callback=calibration_rcv_callback,
  ),
  ProcessConfig(
    proc_name="dmonitoringd",
    pub_sub={
      "driverState": ["dMonitoringState"],
      "liveCalibration": [], "carState": [], "model": [], "gpsLocation": [],
    },
    ignore=[("logMonoTime", 0), ("valid", True)],
    init_callback=get_car_params,
    should_recv_callback=None,
  ),
]

def replay_process(cfg, lr):
  sub_sockets = [s for _, sub in cfg.pub_sub.items() for s in sub]
  pub_sockets = [s for s in cfg.pub_sub.keys() if s != 'can']

  fsm = FakeSubMaster(pub_sockets)
  fpm = FakePubMaster(sub_sockets)
  args = (fsm, fpm)
  if 'can' in list(cfg.pub_sub.keys()):
    can_sock = FakeSocket()
    args = (fsm, fpm, can_sock)

  all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)
  pub_msgs = [msg for msg in all_msgs if msg.which() in list(cfg.pub_sub.keys())]

  params = Params()
  params.clear_all()
  params.manager_start()
  params.put("OpenpilotEnabledToggle", "1")
  params.put("Passive", "0")
  params.put("CommunityFeaturesToggle", "1")

  os.environ['NO_RADAR_SLEEP'] = "1"
  manager.prepare_managed_process(cfg.proc_name)
  mod = importlib.import_module(manager.managed_processes[cfg.proc_name])
  thread = threading.Thread(target=mod.main, args=args)
  thread.daemon = True
  thread.start()

  if cfg.init_callback is not None:
    if 'can' not in list(cfg.pub_sub.keys()):
      can_sock = None
    cfg.init_callback(all_msgs, fsm, can_sock)

  CP = car.CarParams.from_bytes(params.get("CarParams", block=True))

  # wait for started process to be ready
  if 'can' in list(cfg.pub_sub.keys()):
    can_sock.wait_for_recv()
  else:
    fsm.wait_for_update()

  log_msgs, msg_queue = [], []
  for msg in tqdm(pub_msgs):
    if cfg.should_recv_callback is not None:
      recv_socks, should_recv = cfg.should_recv_callback(msg, CP, cfg, fsm)
    else:
      recv_socks = [s for s in cfg.pub_sub[msg.which()] if
                      (fsm.frame + 1) % int(service_list[msg.which()].frequency / service_list[s].frequency) == 0]
      should_recv = bool(len(recv_socks))

    if msg.which() == 'can':
      can_sock.send(msg.as_builder().to_bytes())
    else:
      msg_queue.append(msg.as_builder())

    if should_recv:
      fsm.update_msgs(0, msg_queue)
      msg_queue = []

      recv_cnt = len(recv_socks)
      while recv_cnt > 0:
        m = fpm.wait_for_msg()
        log_msgs.append(m)

        recv_cnt -= m.which() in recv_socks
  return log_msgs
