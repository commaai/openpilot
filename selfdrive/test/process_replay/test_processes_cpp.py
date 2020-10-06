import os
import threading
import time
import subprocess

if "CI" in os.environ:
  def tqdm(x):
    return x
else:
  from tqdm import tqdm   # type: ignore

import cereal.messaging as messaging
from collections import namedtuple
from tools.lib.logreader import LogReader
from selfdrive.test.process_replay.test_processes import get_segment
from common.basedir import BASEDIR

ProcessConfig = namedtuple('ProcessConfig', ['proc_name', 'pub_sub', 'ignore', 'command', 'path', 'segment', 'wait_for_response'])

CONFIGS = [
  ProcessConfig(
    proc_name="ubloxd",
    pub_sub={
      "ubloxRaw": ["ubloxGnss", "gpsLocationExternal"],
    },
    ignore=[],
    command="./ubloxd",
    path="selfdrive/locationd/",
    segment="0375fdf7b1ce594d|2019-06-13--08-32-25--3",
    wait_for_response=True
  ),
]
def launcher(arg, cwd):
  os.chdir(os.path.join(BASEDIR, cwd))
  subprocess.Popen(arg, stderr=subprocess.PIPE, shell=True) 

def replay_process(config, logreader):
  pub_sockets = [s for s in config.pub_sub.keys()]  # We dump data from logs here
  sub_sockets = [s for _, sub in config.pub_sub.items() for s in sub]  # We get responses here
  pm = messaging.PubMaster(pub_sockets)
  sm = messaging.SubMaster(sub_sockets)

  print("Sorting logs")
  all_msgs = sorted(logreader, key=lambda msg: msg.logMonoTime)
  pub_msgs = [msg for msg in all_msgs if msg.which() in list(config.pub_sub.keys())]

  thread = threading.Thread(target=launcher, args=(config.command, config.path))
  thread.daemon = True
  thread.start()

  time.sleep(5)  # We give the process time to start
  # for msg in tqdm(pub_msgs):
  #   pm.send(msg.which(), msg.as_builder())
  #   if config.wait_for_response:
  #     sm.update(100)
  log_msgs= []
  for msg in tqdm(pub_msgs):
    pm.send(msg.which(), msg.as_builder())
    sm.update(100)
  return log_msgs

def test_config():
  for cfg in CONFIGS:
    URL = cfg.segment
    lr = LogReader(get_segment(URL))
    replay_process(cfg, lr)


if __name__ == "__main__":
  test_config()
