import os
import time
import subprocess
import signal
from hashlib import sha256

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

ProcessConfig_cpp = namedtuple('ProcessConfig_cpp', ['proc_name', 'pub_sub', 'ignore', 'command', 'path', 'segment', 'wait_for_response'])

CONFIGS = [
  ProcessConfig_cpp(
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

def replay_process(config, logreader):
  pub_sockets = [s for s in config.pub_sub.keys()]  # We dump data from logs here
  sub_sockets = [s for _, sub in config.pub_sub.items() for s in sub]  # We get responses here
  pm = messaging.PubMaster(pub_sockets)
  sm = messaging.SubMaster(sub_sockets)

  print("Sorting logs")
  all_msgs = sorted(logreader, key=lambda msg: msg.logMonoTime)
  pub_msgs = [msg for msg in all_msgs if msg.which() in list(config.pub_sub.keys())]
  os.chdir(os.path.join(BASEDIR, config.path))
  p = subprocess.Popen(config.command, stderr=subprocess.PIPE)

  time.sleep(5)  # We give the process time to start
  #log_msgs= []
  #for msg in tqdm(pub_msgs):
  #  pm.send(msg.which(), msg.as_builder())
  #  sm.update(100)
  #  for s in sub_sockets:
  #    if sm.updated[s]:
  #      log_msgs.append(str(sm.__getitem__(s)))
  log_msgs = []
  for msg in tqdm(pub_msgs):
    pm.send(msg.which(), msg.as_builder())
    for s in sub_sockets:
      if sm.updated[s]:
        log_msgs.append(sm.__getitem__(s))
      else:
        sm.update()
  os.kill(p.pid, signal.SIGINT)
  return log_msgs

def test_config():
  for cfg in CONFIGS:
    URL = cfg.segment
    lr = LogReader(get_segment(URL))
    response = replay_process(cfg, lr)
    dump = "".join(list(map(str,response)))
    hs = sha256(dump.encode('utf-8')).hexdigest()
    print(hs)
    print(len(response))
    #print(response[0])
if __name__ == "__main__":
  #test_config()
  path = "https://commadataci.blob.core.windows.net/openpilotci/0375fdf7b1ce594d|2019-06-13--08-32-25--3_controlsd_925721984d73b0751e18001f2bacdca9832d9278.bz2"
  lr = LogReader(path)
  print(len(list(lr)))
