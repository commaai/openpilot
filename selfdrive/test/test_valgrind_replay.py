import os
import threading
import time
import unittest
import subprocess

if "CI" in os.environ:
  def tqdm(x):
    return x
else:
  from tqdm import tqdm   # type: ignore

import cereal.messaging as messaging
from collections import namedtuple
from tools.lib.logreader import LogReader

ProcessConfig = namedtuple('ProcessConfig', ['proc_name', 'pub_sub', 'ignore', 'command', 'path'])

BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"

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

class SimplePubMaster():
  def __init__(self, services):  # pylint: disable=super-init-not-called
    self.sock = {}
    for s in services:
      self.sock[s] = messaging.pub_sock(s)

  def send(self, s, dat):
    # print(dat)
    self.sock[s].send(dat.to_bytes())

class TestValgrind(unittest.TestCase):

  def valgrindlauncher(self, arg, cwd):
    os.chdir(cwd)

    # Run valgrind on a process
    command = "valgrind --leak-check=full " + arg

    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    _, err = p.communicate()
    error_msg = str(err, encoding='utf-8')
    err_lost1 = error_msg.split("definitely lost: ")[1]
    err_lost2 = error_msg.split("indirectly lost: ")[1]
    err_lost3 = error_msg.split("possibly lost: ")[1]
    definitely_lost_amount = int(err_lost1.split(" ")[0])
    indirectly_lost_amount = int(err_lost2.split(" ")[0])
    possibly_lost_amount = int(err_lost3.split(" ")[0])
    if max(definitely_lost_amount, indirectly_lost_amount, possibly_lost_amount) > 0:
      self.leak = True
      print(err)
      return
    self.leak = False

  def replay_process(self, cfg, lr):
    pub_sockets = [s for s in cfg.pub_sub.keys() if s != 'can']  # We dump data from logs here

    pm = SimplePubMaster(pub_sockets)
    print("Sorting logs")
    all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)
    pub_msgs = [msg for msg in all_msgs if msg.which() in list(cfg.pub_sub.keys())]

    thread = threading.Thread(target=self.valgrindlauncher, args=(cfg.command, cfg.path))
    thread.daemon = True
    thread.start()
    time.sleep(5)  # We give the process time to start
    for msg in tqdm(pub_msgs):
      pm.send(msg.which(), msg.as_builder())

  def get_segment(self, segment_name, original=True):
    route_name, segment_num = segment_name.rsplit("--", 1)
    if original:
      rlog_url = BASE_URL + "%s/%s/rlog.bz2" % (route_name.replace("|", "/"), segment_num)
    else:
      process_replay_dir = os.path.dirname(os.path.abspath(__file__))
      model_ref_commit = open(os.path.join(process_replay_dir, "model_ref_commit")).read().strip()
      rlog_url = BASE_URL + "%s/%s/rlog_%s.bz2" % (route_name.replace("|", "/"), segment_num, model_ref_commit)

    return rlog_url

  def test_config_0(self):
    cfg = CONFIGS[0]

    URL = self.get_segment("0375fdf7b1ce594d|2019-06-13--08-32-25--3")
    lr = LogReader(URL)
    self.replay_process(cfg, lr)
    # Wait for the replay to complete
    time.sleep(30)
    assert not self.leak

if __name__ == "__main__":
  unittest.main()
