#!/usr/bin/env python3
import glob
import os
import shutil
import time
import unittest

import cereal.messaging as messaging
from selfdrive.manager.process_config import managed_processes
from system.swaglog import cloudlog, SWAGLOG_DIR


class TestLogmessaged(unittest.TestCase):

  def setUp(self):
    if os.path.exists(SWAGLOG_DIR):
      shutil.rmtree(SWAGLOG_DIR)

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
    return list(glob.glob(os.path.join(SWAGLOG_DIR, "swaglog.*")))

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
