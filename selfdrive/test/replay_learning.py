import os
import sys
import threading
import time

if "CI" in os.environ:
  def tqdm(x):
    return x
else:
  from tqdm import tqdm   # type: ignore

import cereal.messaging as messaging
from collections import namedtuple
from tools.lib.logreader import LogReader

ProcessConfig = namedtuple('ProcessConfig', ['proc_name', 'pub_sub', 'ignore', 'command', 'path'])

def wait_for_event(evt):
  if not evt.wait(20):
    if threading.currentThread().getName() == "MainThread":
      # tested process likely died. don't let test just hang
      raise Exception("Timeout reached. Tested process likely crashed.")
    else:
      # done testing this process, let it die
      sys.exit(0)

class SimplePubMaster():
  def __init__(self, services):  # pylint: disable=super-init-not-called
    self.sock = {}
    for s in services:
      self.sock[s] = messaging.pub_sock(s)

  def send(self, s, dat):
    # print(dat)
    self.sock[s].send(dat.to_bytes())


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
  recv_socks = []
  frame = fsm.frame + 1  # incrementing hasn't happened yet in SubMaster
  if frame == 0 or (msg.which() == 'cameraOdometry' and (frame % 5) == 0):
    recv_socks = ["liveCalibration"]
  return recv_socks, fsm.frame == 0 or msg.which() == 'cameraOdometry'

CONFIGS = [
  ProcessConfig(
    proc_name="ubloxd",
    pub_sub={
      "ubloxRaw": ["ubloxGnss", "gpsLocationExternal"],
    },
    ignore=[],
    command="./ubloxd & sleep 10; kill $!",
    path="../locationd",
  ),
  ProcessConfig(
    proc_name="loggerd",
    pub_sub={
      "ubloxRaw": ["ubloxGnss", "gpsLocationExternal"],
    },
    ignore=[],
    command="./loggerd & sleep 10; kill $!",
    path="../loggerd",
  ),
]

def valgrindlauncher(arg, cwd):
  os.chdir(cwd)

  # Run valgrind on a process
  command = "valgrind --leak-check=full " + arg
  print(command)
  output = os.popen(command)
  while True:
    s = output.read()
    if s == "":
      break
    print(s)

def replay_process(cfg, lr):
  pub_sockets = [s for s in cfg.pub_sub.keys() if s != 'can']  # We dump data from logs here

  pm = SimplePubMaster(pub_sockets)
  print("Sorting logs")
  all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)
  pub_msgs = [msg for msg in all_msgs if msg.which() in list(cfg.pub_sub.keys())]

  thread = threading.Thread(target=valgrindlauncher, args=(cfg.command, cfg.path))
  thread.daemon = True
  thread.start()
  time.sleep(10)  # We give the process time to start
  for msg in tqdm(pub_msgs):
    pm.send(msg.which(), msg.as_builder())

  # for s in pub_sockets:
  #   pm.send(s, b"")

URL = "https://commadata2.blob.core.windows.net/commadata2/a74b011b32b51b56/2020-09-21--10-29-15\
/0/rlog.bz2?se=2020-09-29T13%3A13%3A40Z&sp=r&sv=2018-03-28&sr=b&rscd=attachment%3B%20filename%3Da74b011b32b51b56_2020-09-21--10-29-15--0--rlog.bz2&sig=iPiuSfthiYo3xL3tDKoKwr8kQH7odHW%2BqGcCqmnY2EY%3D"
if __name__ == "__main__":
  cfg = CONFIGS[0]

  lr = LogReader(URL)
  print(str(cfg))
  replay_process(cfg, lr)
  time.sleep(15)
