#!/usr/bin/env python3
import glob
import os
import time
import unittest

import cereal.messaging as messaging
from openpilot.selfdrive.manager.process_config import managed_processes
from openpilot.system.swaglog import cloudlog, ipchandler
from selfdrive.test.helpers import temporary_swaglog_dir


class TestLogmessaged(unittest.TestCase):
  def _setup(self, temp_dir):
    # clear the IPC buffer in case some other tests used cloudlog and filled it
    ipchandler.close()
    ipchandler.connect()

    self.temp_dir = temp_dir
    managed_processes['logmessaged'].start()
    self.sock = messaging.sub_sock("logMessage", timeout=1000, conflate=False)
    self.error_sock = messaging.sub_sock("logMessage", timeout=1000, conflate=False)

    # ensure sockets are connected
    time.sleep(0.2)
    messaging.drain_sock(self.sock)
    messaging.drain_sock(self.error_sock)

  def tearDown(self):
    del self.sock
    del self.error_sock
    managed_processes['logmessaged'].stop(block=True)

  def _get_log_files(self):
    return list(glob.glob(os.path.join(self.temp_dir, "swaglog.*")))

  @temporary_swaglog_dir
  def test_simple_log(self, temp_dir):
    self._setup(temp_dir)
    msgs = [f"abc {i}" for i in range(10)]
    for m in msgs:
      cloudlog.error(m)
    time.sleep(3)
    m = messaging.drain_sock(self.sock)
    assert len(m) == len(msgs)
    assert len(self._get_log_files()) >= 1

  @temporary_swaglog_dir
  def test_big_log(self, temp_dir):
    self._setup(temp_dir)
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
