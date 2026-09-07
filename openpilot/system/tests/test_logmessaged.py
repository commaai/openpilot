from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time
from unittest import mock

import openpilot.cereal.messaging as messaging
from openpilot.common.file_queue import FileQueue
from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog, ipchandler
from openpilot.common.test import OpenpilotTestCase
from openpilot.system.manager.process_config import managed_processes

NATIVE = Path(__file__).resolve().parents[2] / 'common/tests/test_swaglog'


def crash_before_publish():
  with mock.patch('os.rename', side_effect=lambda *args: os._exit(0)):
    cloudlog.info('unpublished')


class TestLogmessaged(OpenpilotTestCase):
  def setup_method(self):
    ipchandler.close()
    ipchandler.connect()
    self.queue = FileQueue(Paths.swaglog_ipc())

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
    assert not messaging.drain_sock(sock)

  def test_burst_capacity(self):
    message = 'a' * (8 * 1024 * 1024)

    def produce(native):
      if native:
        subprocess.run([str(NATIVE), '--emit', '4'], input=message.encode(), check=True, timeout=30)
      else:
        for _ in range(4):
          cloudlog.info(message)

    def usage():
      sizes = {}  # Deduplicate files that move between directories during the scan.
      for directory in (self.queue.pending, self.queue.ready):
        for path in directory.iterdir():
          try:
            sizes[path.name] = FileQueue._charge(path.stat().st_size)
          except FileNotFoundError:
            pass
      return sum(sizes.values())

    with ThreadPoolExecutor(max_workers=8) as pool:
      writers = [pool.submit(produce, i % 2) for i in range(8)]
      while not all(writer.done() for writer in writers):
        assert usage() <= 2 * self.queue.capacity
        time.sleep(0.001)
      for writer in writers:
        writer.result()
    assert self.queue.capacity // 2 < usage() <= 2 * self.queue.capacity
    while (data := self.queue.receive()) is not None:
      assert json.loads(data[1:])['msg'] == message
    assert usage() == 0
    cloudlog.info('x' * self.queue.capacity)
    assert self.queue.receive() is None
    cloudlog.info('after overflow')
    assert json.loads(self.queue.receive()[1:])['msg'] == 'after overflow'

  def test_crashed_writer(self):
    process = multiprocessing.get_context('fork').Process(target=crash_before_publish)
    process.start()
    try:
      process.join(10)
      assert process.exitcode == 0
      assert list(self.queue.pending.iterdir())
      cloudlog.info('committed')
      assert json.loads(self.queue.receive()[1:])['msg'] == 'committed'
      assert self.queue.receive() is None
      assert not list(self.queue.pending.iterdir())
    finally:
      if process.is_alive():
        process.kill()
        process.join()
