import sys
import json
import time
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from openpilot.common.hardware.hw import Paths
import openpilot.cereal.messaging as messaging
from openpilot.common.shm_queue import ShmQueue
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.swaglog import cloudlog, ipchandler
from openpilot.system.manager.process_config import managed_processes

NATIVE = Path(__file__).resolve().parents[2] / 'common/tests/test_swaglog'


class TestLogmessaged(OpenpilotTestCase):
  def setup_method(self):
    ipchandler.close()
    ipchandler.connect()
    self.queue = ShmQueue(Paths.swaglog_ipc())

  def teardown_method(self):
    managed_processes['logmessaged'].stop(block=True)
    ipchandler.close()

  def wait_for(self, condition):
    deadline = time.monotonic() + 10
    while not condition():
      assert time.monotonic() < deadline, 'timed out waiting for logs'
      time.sleep(0.01)

  def logsize(self):
    return sum(p.stat().st_size for p in Path(Paths.swaglog_root()).glob('swaglog.*'))

  def test_publication(self):
    sockets = {s: messaging.sub_sock(s, conflate=False) for s in ('logMessage', 'errorLogMessage')}
    received = {s: [] for s in sockets}
    managed_processes['logmessaged'].start()

    def collect():
      cloudlog.error('hello')  # Retry until both subscribers connect.
      for service, sock in sockets.items():
        received[service].extend(json.loads(getattr(m, service))['msg'] for m in messaging.drain_sock(sock))
      return all('hello' in records for records in received.values())

    self.wait_for(collect)
    self.wait_for(lambda: self.logsize() > 0)

  def test_large_logs(self):
    message = 'a' * (3 * 1024 * 1024)
    for _ in range(10):
      cloudlog.info(message)
    sock = messaging.sub_sock('logMessage', conflate=False)
    managed_processes['logmessaged'].start()
    self.wait_for(lambda: self.logsize() > 10 * len(message))
    assert self.logsize() < 10 * (len(message) + 1024)
    assert all(json.loads(m.logMessage)['msg'] != message for m in messaging.drain_sock(sock))

  def test_burst_capacity(self):
    message = 'a' * (8 * 1024 * 1024)

    with ThreadPoolExecutor(max_workers=4) as pool:
      writers = [pool.submit(subprocess.run, [str(NATIVE), '--emit', '4'],
                             input=message.encode(), check=True, timeout=30) for _ in range(4)]
      for _ in range(4):
        cloudlog.info(message)
      for writer in writers:
        writer.result()
    # No consumer ran during the burst, so accepted files retain any overfill.
    usage = sum(p.stat().st_blocks * 512 + ShmQueue.PAGE_SIZE for p in self.queue.ready.iterdir())
    assert self.queue.capacity // 2 < usage <= 2 * self.queue.capacity
    # Fits individually, but must be dropped because the spool is already over half full.
    cloudlog.info('x' * (self.queue.capacity // 2))
    while (data := self.queue.receive()) is not None:
      assert json.loads(data[1:])['msg'] == message
    assert not list(self.queue.pending.iterdir())
    assert not list(self.queue.ready.iterdir())
    cloudlog.info('x' * self.queue.capacity)
    assert self.queue.receive() is None
    cloudlog.info('after overflow')
    data = self.queue.receive()
    assert data is not None
    assert json.loads(data[1:])['msg'] == 'after overflow'

  def test_crashed_writer(self):
    script = """
import os
from openpilot.common.swaglog import cloudlog
os.rename = lambda *args: os._exit(0)
cloudlog.info('unpublished')
"""
    subprocess.run([sys.executable, '-c', script], check=True, timeout=10)
    assert list(self.queue.pending.iterdir())
    cloudlog.info('committed')
    data = self.queue.receive()
    assert data is not None
    assert json.loads(data[1:])['msg'] == 'committed'
    assert self.queue.receive() is None
    assert not list(self.queue.pending.iterdir())
