import os
import sys
import json
import time
import subprocess
from pathlib import Path
from unittest.mock import patch
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

  def test_slots(self):
    with patch.object(ShmQueue, 'MAX_MESSAGE_SIZE', 1):
      assert not self.queue.send(b'xx')
    def native_burst():
      subprocess.run([str(NATIVE), '--emit', '100'], input=b'native', check=True, timeout=30)

    with ThreadPoolExecutor(max_workers=4) as pool:
      writers = [pool.submit(native_burst) for _ in range(4)]
      with patch('os.scandir', side_effect=AssertionError('producer scanned the backlog')):
        for _ in range(100):
          cloudlog.info('python')
      for writer in writers:
        writer.result()
    assert len(list(self.queue.ready.iterdir())) == len(list(self.queue.slots.iterdir())) <= ShmQueue.SLOT_COUNT
    messages = []
    while (data := self.queue.receive()) is not None:
      messages.append(json.loads(data[1:])['msg'])
    assert set(messages) == {'python', 'native'}
    assert not list(self.queue.pending.iterdir())
    assert not list(self.queue.slots.iterdir())

    # Live claims occupy every slot without allocating thousands of payload files.
    for slot in range(ShmQueue.SLOT_COUNT):
      (self.queue.slots / str(slot)).symlink_to(f'{0:020d}-{os.getpid()}-1-{slot}')
    with patch('os.scandir', side_effect=AssertionError('producer scanned the backlog')), \
         patch('openpilot.common.shm_queue.random.randrange', return_value=0) as choose:
      cloudlog.info('full')
    assert choose.call_count == ShmQueue.CLAIM_ATTEMPTS
    subprocess.run([str(NATIVE), '--emit'], input=b'full', check=True, timeout=10)
    assert not list(self.queue.ready.iterdir())
    assert not list(self.queue.pending.iterdir())
    for slot in self.queue.slots.iterdir():
      slot.unlink()
    cloudlog.info('reused')
    data = self.queue.receive()
    assert data is not None
    assert json.loads(data[1:])['msg'] == 'reused'
    assert not list(self.queue.slots.iterdir())

  def test_crashed_writer(self):
    for crash in ('Path.open', 'os.rename'):
      with self.subTest(crash=crash):
        script = f"""
import os
from pathlib import Path
from openpilot.common.swaglog import cloudlog
{crash} = lambda *args: os._exit(0)
cloudlog.info('unpublished')
"""
        subprocess.run([sys.executable, '-c', script], check=True, timeout=10)
        assert list(self.queue.slots.iterdir())
        cloudlog.info('committed')
        data = self.queue.receive()
        assert data is not None
        assert json.loads(data[1:])['msg'] == 'committed'
        assert self.queue.receive() is None
        assert not list(self.queue.pending.iterdir())
        assert not list(self.queue.slots.iterdir())

    # Reader died after releasing a slot but before removing its published file.
    with patch('openpilot.common.shm_queue.random.randrange', return_value=0):
      cloudlog.info('old')
      old = next(self.queue.ready.iterdir())
      self.queue._release(self.queue.slots / '0', old.name)
      cloudlog.info('new')
    data = self.queue.receive()
    assert data is not None
    assert json.loads(data[1:])['msg'] == 'old'
    assert (self.queue.slots / '0').is_symlink()
    data = self.queue.receive()
    assert data is not None
    assert json.loads(data[1:])['msg'] == 'new'
    assert not list(self.queue.slots.iterdir())
