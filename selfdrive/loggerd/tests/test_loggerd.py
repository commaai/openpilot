#!/usr/bin/env python3
import os
import random
import shutil
import string
import subprocess
import time
import unittest
from pathlib import Path
from tqdm import trange

from common.basedir import BASEDIR
from common.timeout import Timeout
#from selfdrive.test.helpers import with_processes
from tools.lib.logreader import LogReader

class TestLoggerd(unittest.TestCase):

  def _get_log_dir(self, x):
    for p in x.split(' '):
      path = Path(p.strip())
      if path.is_dir():
        return path
    return None
  
  def test_bootlog(self):
    launch_log = ''.join([str(random.choice(string.printable)) for _ in range(100)])
    with open("/tmp/launch_log", "w") as f:
      f.write(launch_log)

    # generate bootlog
    with Timeout(5):
      out = subprocess.check_output(["./loggerd", "--bootlog"], cwd=os.path.join(BASEDIR, "selfdrive/loggerd"), encoding='utf-8')

    # check existence
    d = self._get_log_dir(out) 
    bootlog_path = Path(os.path.join(d, "bootlog.bz2"))
    assert bootlog_path.is_file(), "failed to create bootlog file"
    
    # check msgs
    lr = list(LogReader(str(bootlog_path)))
    assert len(lr) == 4 # boot + initData + 2x sentinel
    bootlog_msgs = [m for m in lr if m.which() == 'boot']
    assert len(bootlog_msgs) == 1

    # sanity check values
    boot = bootlog_msgs.pop().boot
    assert abs(boot.wallTimeNanos - time.time_ns()) < 5*1e9 # within 5s
    assert boot.launchLog == launch_log

    for field, path in [("lastKmsg", "console-ramoops"), ("lastPmsg", "pmsg-ramoops-0")]:
      path = Path(os.path.join("/sys/fs/pstore/", path))
      val = b""
      if path.is_file():
        val = open(path).read("rb")
      assert getattr(boot, field) == val

if __name__ == "__main__":
  unittest.main()
