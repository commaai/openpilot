import os
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

class SimplePubMaster():
  def __init__(self, services):  # pylint: disable=super-init-not-called
    self.sock = {}
    for s in services:
      self.sock[s] = messaging.pub_sock(s)

  def send(self, s, dat):
    # print(dat)
    self.sock[s].send(dat.to_bytes())


CONFIGS = [
  ProcessConfig(
    proc_name="ubloxd",
    pub_sub={
      "ubloxRaw": ["ubloxGnss", "gpsLocationExternal"],
    },
    ignore=[],
    command="./ubloxd & sleep 20; kill $!",
    path="../locationd",
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
  print(len(pub_msgs))
  thread = threading.Thread(target=valgrindlauncher, args=(cfg.command, cfg.path))
  thread.daemon = True
  thread.start()
  time.sleep(5)  # We give the process time to start
  for msg in tqdm(pub_msgs):
    pm.send(msg.which(), msg.as_builder())

BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"

def get_segment(segment_name, original=True):
  route_name, segment_num = segment_name.rsplit("--", 1)
  if original:
    rlog_url = BASE_URL + "%s/%s/rlog.bz2" % (route_name.replace("|", "/"), segment_num)
  else:
    process_replay_dir = os.path.dirname(os.path.abspath(__file__))
    model_ref_commit = open(os.path.join(process_replay_dir, "model_ref_commit")).read().strip()
    rlog_url = BASE_URL + "%s/%s/rlog_%s.bz2" % (route_name.replace("|", "/"), segment_num, model_ref_commit)

  return rlog_url

if __name__ == "__main__":
  cfg = CONFIGS[0]

  URL = get_segment("0375fdf7b1ce594d|2019-06-13--08-32-25--3")
  print(URL)
  lr = LogReader(URL)
  print(str(cfg))
  replay_process(cfg, lr)
  time.sleep(30)
