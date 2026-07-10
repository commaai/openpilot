import glob
import os
import time

import openpilot.cereal.messaging as messaging
from openpilot.system.manager.process_config import managed_processes
from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog, ipchandler


class TestLogmessaged:
  def setup_method(self):
    # clear the IPC buffer in case some other tests used cloudlog and filled it
    ipchandler.close()
    ipchandler.connect()

    managed_processes['logmessaged'].start()
    self.sock = messaging.sub_sock("logMessage", timeout=1000, conflate=False)
    self.error_sock = messaging.sub_sock("logMessage", timeout=1000, conflate=False)

    # ensure sockets are connected
    time.sleep(0.5)
    messaging.drain_sock(self.sock)
    messaging.drain_sock(self.error_sock)

  def teardown_method(self):
    del self.sock
    del self.error_sock
    managed_processes['logmessaged'].stop(block=True)

  def _get_log_files(self):
    return list(glob.glob(os.path.join(Paths.swaglog_root(), "swaglog.*")))

  def test_simple_log(self):
    msgs = [f"abc {i}" for i in range(10)]
    for m in msgs:
      cloudlog.error(m)
    time.sleep(0.5)
    m = messaging.drain_sock(self.sock)
    assert len(m) == len(msgs)
    assert len(self._get_log_files()) >= 1

  def test_big_log(self):
    n = 10
    msg = "a"*3*1024*1024
    for _ in range(n):
      cloudlog.info(msg)
    time.sleep(0.5)

    msgs = messaging.drain_sock(self.sock)
    assert len(msgs) == 0

    logsize = sum([os.path.getsize(f) for f in self._get_log_files()])
    assert (n*len(msg)) < logsize < (n*(len(msg)+1024))

  def test_medium_log(self):
    # records between the msgq limit (a message must fit in a third of the
    # 256KiB queue, ~85KB) and the old 2MB guard used to crash logmessaged
    # with a msgq assertion, killing logging for the rest of the drive
    big = "a" * (200 * 1024)
    cloudlog.info(big)
    cloudlog.error("alive after big record")
    time.sleep(0.5)

    # logmessaged survived the big record
    assert managed_processes['logmessaged'].proc.is_alive()

    # the big record is dropped from the socket, later records still flow
    msgs = messaging.drain_sock(self.sock)
    texts = [m.logMessage for m in msgs]
    assert not any(len(t) >= len(big) for t in texts)
    assert any("alive after big record" in t for t in texts)

    # both records still land in the on-disk log
    logsize = sum([os.path.getsize(f) for f in self._get_log_files()])
    assert logsize > len(big)

