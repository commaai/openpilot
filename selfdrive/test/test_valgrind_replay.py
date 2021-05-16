import os
import threading
import time
import unittest
import subprocess
import signal

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


class TestValgrind(unittest.TestCase):
  def extract_leak_sizes(self, log):
    if "All heap blocks were freed -- no leaks are possible" in log:
      return (0,0,0)

    log = log.replace(",","")  # fixes casting to int issue with large leaks
    err_lost1 = log.split("definitely lost: ")[1]
    err_lost2 = log.split("indirectly lost: ")[1]
    err_lost3 = log.split("possibly lost: ")[1]
    definitely_lost = int(err_lost1.split(" ")[0])
    indirectly_lost = int(err_lost2.split(" ")[0])
    possibly_lost = int(err_lost3.split(" ")[0])
    return (definitely_lost, indirectly_lost, possibly_lost)

  def valgrindlauncher(self, arg, cwd):
    os.chdir(os.path.join(BASEDIR, cwd))
    # Run valgrind on a process
    command = "valgrind --leak-check=full " + arg
    p = subprocess.Popen(command, stderr=subprocess.PIPE, shell=True, preexec_fn=os.setsid)  # pylint: disable=W1509
    while not self.done:
      time.sleep(0.1)

    os.killpg(os.getpgid(p.pid), signal.SIGINT)
    _, err = p.communicate()
    error_msg = str(err, encoding='utf-8')
    with open(os.path.join(BASEDIR, "selfdrive/test/valgrind_logs.txt"), "a") as f:
      f.write(error_msg)
      f.write(5 * "\n")
    definitely_lost, indirectly_lost, possibly_lost = self.extract_leak_sizes(error_msg)
    if max(definitely_lost, indirectly_lost, possibly_lost) > 0:
      self.leak = True
      print("LEAKS from", arg, "\nDefinitely lost:", definitely_lost, "\nIndirectly lost", indirectly_lost, "\nPossibly lost", possibly_lost)
    else:
      self.leak = False

  def replay_process(self, config, logreader):
    pub_sockets = [s for s in config.pub_sub.keys()]  # We dump data from logs here
    sub_sockets = [s for _, sub in config.pub_sub.items() for s in sub]  # We get responses here
    pm = messaging.PubMaster(pub_sockets)
    sm = messaging.SubMaster(sub_sockets)

    print("Sorting logs")
    all_msgs = sorted(logreader, key=lambda msg: msg.logMonoTime)
    pub_msgs = [msg for msg in all_msgs if msg.which() in list(config.pub_sub.keys())]

    thread = threading.Thread(target=self.valgrindlauncher, args=(config.command, config.path))
    thread.daemon = True
    thread.start()

    while not all(pm.all_readers_updated(s) for s in config.pub_sub.keys()):
      time.sleep(0)

    for msg in tqdm(pub_msgs):
      pm.send(msg.which(), msg.as_builder())
      if config.wait_for_response:
        sm.update(100)

    self.done = True

  def test_config(self):
    open(os.path.join(BASEDIR, "selfdrive/test/valgrind_logs.txt"), "w").close()

    for cfg in CONFIGS:
      self.done = False
      URL = cfg.segment
      lr = LogReader(get_segment(URL))
      self.replay_process(cfg, lr)
      time.sleep(1)  # Wait for the logs to get written
      self.assertFalse(self.leak)

if __name__ == "__main__":
  unittest.main()
