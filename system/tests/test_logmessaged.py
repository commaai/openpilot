#!/usr/bin/env python3
import glob
import os
import time
import unittest

import cereal.messaging as messaging
from openpilot.selfdrive.manager.process_config import managed_processes
from openpilot.system.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog, ipchandler


class TestLogmessaged(unittest.TestCase):
  def setUp(self):
    # clear the IPC buffer in case some other tests used cloudlog and filled it
    ipchandler.close()
    ipchandler.connect()

    managed_processes['logmessaged'].start()
    self.sock = messaging.sub_sock("logMessage", timeout=1000, conflate=False)
    self.error_sock = messaging.sub_sock("logMessage", timeout=1000, conflate=False)

    # ensure sockets are connected
    time.sleep(1)
    messaging.drain_sock(self.sock)
    messaging.drain_sock(self.error_sock)

  def tearDown(self):
    del self.sock
    del self.error_sock
    managed_processes['logmessaged'].stop(block=True)

  def _get_log_files(self):
    return list(glob.glob(os.path.join(Paths.swaglog_root(), "swaglog.*")))

  def test_simple_log(self):
    msgs = [f"abc {i}" for i in range(10)]
    for m in msgs:
      cloudlog.error(m)
    time.sleep(3)
    m = messaging.drain_sock(self.sock)
    assert len(m) == len(msgs)
    assert len(self._get_log_files()) >= 1

  def test_big_log(self):
    n = 10
    msg = "a"*3*1024*1024
    for _ in range(n):
      cloudlog.info(msg)
    time.sleep(3)

    msgs = messaging.drain_sock(self.sock)
    assert len(msgs) == 0

    logsize = sum([os.path.getsize(f) for f in self._get_log_files()])
    assert (n*len(msg)) < logsize < (n*(len(msg)+1024))


if __name__ == "__main__":
  unittest.main()
