#!/usr/bin/env python2
import gc
import os
import time

if "CI" in os.environ:
  tqdm = lambda x: x
else:
  from tqdm import tqdm

from cereal import car
from selfdrive.car.car_helpers import get_car
import selfdrive.manager as manager
import selfdrive.messaging as messaging
from common.params import Params
from selfdrive.services import service_list
from collections import namedtuple

ProcessConfig = namedtuple('ProcessConfig', ['proc_name', 'pub_sub', 'ignore', 'init_callback', 'should_recv_callback'])

def fingerprint(msgs, pub_socks, sub_socks):
  print "start fingerprinting"
  manager.prepare_managed_process("logmessaged")
  manager.start_managed_process("logmessaged")

  can = pub_socks["can"]
  logMessage = messaging.sub_sock(service_list["logMessage"].port)

  time.sleep(1)
  messaging.drain_sock(logMessage)

  # controlsd waits for a health packet before fingerprinting
  msg = messaging.new_message()
  msg.init("health")
  pub_socks["health"].send(msg.to_bytes())

  canmsgs = filter(lambda msg: msg.which() == "can", msgs)
  for msg in canmsgs[:200]:
    can.send(msg.as_builder().to_bytes())

    time.sleep(0.005)
    log = messaging.recv_one_or_none(logMessage)
    if log is not None and "fingerprinted" in log.logMessage:
      break
  manager.kill_managed_process("logmessaged")
  print "finished fingerprinting"

def get_car_params(msgs, pub_socks, sub_socks):
  sendcan = pub_socks.get("sendcan", None)
  if sendcan is None:
    sendcan = messaging.pub_sock(service_list["sendcan"].port)
  logcan = sub_socks.get("can", None)
  if logcan is None:
    logcan = messaging.sub_sock(service_list["can"].port)
  can = pub_socks.get("can", None)
  if can is None:
    can = messaging.pub_sock(service_list["can"].port)

  time.sleep(0.5)

  canmsgs = filter(lambda msg: msg.which() == "can", msgs)
  for m in canmsgs[:200]:
    can.send(m.as_builder().to_bytes())
  _, CP = get_car(logcan, sendcan)
  Params().put("CarParams", CP.to_bytes())
  time.sleep(0.5)
  messaging.drain_sock(logcan)

def radar_rcv_callback(msg, CP):
  if msg.which() != "can":
    return []

  # hyundai and subaru don't have radar
  radar_msgs = {"honda": [0x445], "toyota": [0x19f, 0x22f], "gm": [0x475],
                "hyundai": [], "chrysler": [0x2d4], "subaru": []}.get(CP.carName, None)

  if radar_msgs is None:
    raise NotImplementedError

  for m in msg.can:
    if m.src == 1 and m.address in radar_msgs:
      return ["radarState", "liveTracks"]

  return []

def plannerd_rcv_callback(msg, CP):
  if msg.which() in ["model", "radarState"]:
    time.sleep(0.005)
  else:
    time.sleep(0.002)
  return {"model": ["pathPlan"], "radarState": ["plan"]}.get(msg.which(), [])

CONFIGS = [
  ProcessConfig(
    proc_name="controlsd",
    pub_sub={
      "can": ["controlsState", "carState", "carControl", "sendcan"],
      "thermal":  [], "health": [], "liveCalibration": [], "driverMonitoring": [], "plan": [], "pathPlan": []
    },
    ignore=["logMonoTime", "controlsState.startMonoTime", "controlsState.cumLagMs"],
    init_callback=fingerprint,
    should_recv_callback=None,
  ),
  ProcessConfig(
    proc_name="radard",
    pub_sub={
      "can": ["radarState", "liveTracks"],
      "liveParameters":  [], "controlsState":  [], "model":  [],
    },
    ignore=["logMonoTime", "radarState.cumLagMs"],
    init_callback=get_car_params,
    should_recv_callback=radar_rcv_callback,
  ),
  ProcessConfig(
    proc_name="plannerd",
    pub_sub={
      "model": ["pathPlan"], "radarState": ["plan"],
      "carState": [], "controlsState": [], "liveParameters": [],
    },
    ignore=["logMonoTime", "valid", "plan.processingDelay"],
    init_callback=get_car_params,
    should_recv_callback=plannerd_rcv_callback,
  ),
  ProcessConfig(
    proc_name="calibrationd",
    pub_sub={
      "cameraOdometry": ["liveCalibration"]
    },
    ignore=["logMonoTime"],
    init_callback=get_car_params,
    should_recv_callback=None,
  ),
]

def replay_process(cfg, lr):
  gc.disable()  # gc can occasionally cause canparser to timeout

  pub_socks, sub_socks = {}, {}
  for pub, sub in cfg.pub_sub.iteritems():
    pub_socks[pub] = messaging.pub_sock(service_list[pub].port)

    for s in sub:
      sub_socks[s] = messaging.sub_sock(service_list[s].port)

  all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)
  pub_msgs = filter(lambda msg: msg.which() in pub_socks.keys(), all_msgs)

  params = Params()
  params.manager_start()
  params.put("Passive", "0")

  manager.gctx = {}
  manager.prepare_managed_process(cfg.proc_name)
  manager.start_managed_process(cfg.proc_name)
  time.sleep(3)   # Wait for started process to be ready

  if cfg.init_callback is not None:
    cfg.init_callback(all_msgs, pub_socks, sub_socks)

  CP = car.CarParams.from_bytes(params.get("CarParams", block=True))

  log_msgs = []
  for msg in tqdm(pub_msgs):
    if cfg.should_recv_callback is not None:
      recv_socks = cfg.should_recv_callback(msg, CP)
    else:
      recv_socks = cfg.pub_sub[msg.which()]

    pub_socks[msg.which()].send(msg.as_builder().to_bytes())

    if len(recv_socks):
      # TODO: add timeout
      for sock in recv_socks:
        m = messaging.recv_one(sub_socks[sock])

        # make these values fixed for faster comparison
        m_builder = m.as_builder()
        m_builder.logMonoTime = 0
        m_builder.valid = True
        log_msgs.append(m_builder.as_reader())

  gc.enable()
  manager.kill_managed_process(cfg.proc_name)
  return log_msgs

