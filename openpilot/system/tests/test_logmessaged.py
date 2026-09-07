import glob
import json
import multiprocessing
from pathlib import Path
import signal
import struct
import subprocess
from unittest import mock
import os
import time

from openpilot.common.test import OpenpilotTestCase
import openpilot.cereal.messaging as messaging
from openpilot.system.manager.process_config import managed_processes
from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog, ipchandler
from openpilot.common.shm_queue import ShmQueue


SWAGLOG_TEST = Path(__file__).resolve().parents[2] / "common/tests/test_swaglog"


def produce_logs(prefix, count):
  for i in range(count):
    cloudlog.debug(f'{prefix}:{i}')


def crash_during_log(connection):
  copy = ShmQueue._copy

  def partial_copy(queue, pos, data):
    copy(queue, pos, data)
    connection.send(True)
    signal.pause()  # Parent kills us while the log record is incomplete and locked.

  with mock.patch.object(ShmQueue, '_copy', partial_copy):
    cloudlog.debug('incomplete')


class TestLogmessaged(OpenpilotTestCase):
  def setup_method(self):
    # Open the queue in this test's isolated prefix.
    ipchandler.close()
    ipchandler.connect()

    self.sock = messaging.sub_sock("logMessage", timeout=1000, conflate=False)
    self.error_sock = messaging.sub_sock("errorLogMessage", timeout=1000, conflate=False)

  def teardown_method(self):
    ipchandler.close()
    del self.sock
    del self.error_sock
    managed_processes['logmessaged'].stop(block=True)

  def _get_log_files(self):
    return list(glob.glob(os.path.join(Paths.swaglog_root(), "swaglog.*")))

  def test_simple_log(self):
    managed_processes['logmessaged'].start()
    time.sleep(0.5)
    messaging.drain_sock(self.sock)
    messaging.drain_sock(self.error_sock)
    msgs = [f"abc {i}" for i in range(10)]
    for m in msgs:
      cloudlog.error(m)
    time.sleep(0.5)
    m = messaging.drain_sock(self.sock)
    assert len(m) == len(msgs)
    assert len(messaging.drain_sock(self.error_sock)) == len(msgs)
    assert len(self._get_log_files()) >= 1

  def test_big_log(self):
    n = 10
    msg = "a"*3*1024*1024
    for _ in range(n):
      cloudlog.info(msg)
    # Queue the complete burst before starting the reader: contention may drop logs.
    managed_processes['logmessaged'].start()
    time.sleep(0.5)

    msgs = messaging.drain_sock(self.sock)
    assert len(msgs) == 0

    logsize = sum([os.path.getsize(f) for f in self._get_log_files()])
    assert (n*len(msg)) < logsize < (n*(len(msg)+1024))


  def _emit(self, message, native):
    if native:
      subprocess.run([str(SWAGLOG_TEST), '--emit'], input=message.encode(), check=True, timeout=10)
    else:
      cloudlog.debug(message)

  def _reader(self):
    queue = ShmQueue(Paths.swaglog_ipc())
    self.addCleanup(queue.close)
    return queue

  def _read_log(self, reader):
    data = reader.receive()
    self.assertIsNotNone(data)
    self.assertEqual(data[0], 10)
    record = json.loads(data[1:])
    self.assertEqual(record['levelnum'], 10)
    return record['msg']

  def test_swaglog_wrap_and_reopen(self):
    reader = self._reader()
    for native in (False, True):
      for offset in (2, 8):  # Wrap the length prefix, then the payload.
        with self.subTest(native=native, offset=offset):
          self.assertIsNone(reader.receive())
          pos = reader.capacity - offset
          reader.mem[:16] = struct.pack('<QQ', pos, pos)
          message = 'a' * (3 * 1024 * 1024)
          self._emit(message, native)
          ipchandler.close()
          # A newly opened consumer can read the existing producer's record.
          reopened = ShmQueue(Paths.swaglog_ipc())
          try:
            self.assertEqual(self._read_log(reopened), message)
            self.assertIsNone(reopened.receive())
          finally:
            reopened.close()

  def test_swaglog_full_and_oversized(self):
    reader = self._reader()
    message = 'a' * (3 * 1024 * 1024)
    for native in (False, True):
      with self.subTest(native=native):
        # 21 records fit; the next 3 MiB record must be dropped without corruption.
        for _ in range(22):
          self._emit(message, native)
        for _ in range(21):
          self.assertEqual(self._read_log(reader), message)
        self.assertIsNone(reader.receive())
        self._emit('x' * reader.capacity, native)
        self.assertIsNone(reader.receive())
        self._emit('after overflow', native)
        self.assertEqual(self._read_log(reader), 'after overflow')

  def test_swaglog_contention_and_crash(self):
    reader = self._reader()
    cloudlog.debug('committed')
    ctx = multiprocessing.get_context('fork')
    parent, child = ctx.Pipe()
    self.addCleanup(parent.close)
    self.addCleanup(child.close)
    process = ctx.Process(target=crash_during_log, args=(child,))
    process.start()
    try:
      self.assertTrue(parent.poll(10))
      self.assertTrue(parent.recv())
      self._emit('busy', False)
      ipchandler.close()
      self._emit('busy on open', False)
      self._emit('busy on open', True)
    finally:
      process.kill()
      process.join(10)
    self.assertEqual(self._read_log(reader), 'committed')
    self.assertIsNone(reader.receive())
    for native in (False, True):
      self._emit('after crash', native)
      self.assertEqual(self._read_log(reader), 'after crash')

  def test_swaglog_fork(self):
    reader = self._reader()
    # Initialize in the parent first, so children inherit an open producer.
    cloudlog.debug('parent')
    process = multiprocessing.get_context('fork').Process(target=produce_logs, args=('child', 1))
    process.start()
    process.join(10)
    if process.is_alive():
      process.kill()
      process.join()
    self.assertEqual(process.exitcode, 0)
    self.assertEqual(self._read_log(reader), 'parent')
    self.assertEqual(self._read_log(reader), 'child:0')
    subprocess.run([str(SWAGLOG_TEST), '--fork'], check=True, timeout=10)
    self.assertEqual(self._read_log(reader), 'parent')
    self.assertEqual(self._read_log(reader), 'child')
    self.assertIsNone(reader.receive())

  def test_swaglog_concurrent_producers(self):
    reader = self._reader()
    count = 200
    processes = []
    try:
      for i in range(2):
        process = subprocess.Popen([str(SWAGLOG_TEST), '--emit', str(count)], stdin=subprocess.PIPE)
        processes.append(process)
        process.stdin.write(f'native{i}'.encode())
        process.stdin.close()
      for i in range(count):
        cloudlog.debug(f'python:{i}')
      for process in processes:
        self.assertEqual(process.wait(timeout=10), 0)
    finally:
      for process in processes:
        if process.poll() is None:
          process.kill()
          process.wait()
    records = []
    while (data := reader.receive()) is not None:
      self.assertEqual(data[0], 10)
      records.append(json.loads(data[1:])['msg'])
    # Contention can drop records, but accepted records remain whole, unique, and ordered per producer.
    self.assertTrue(records)
    self.assertEqual(len(records), len(set(records)))
    expected = {f'{prefix}:{i}' for prefix in ('python', 'native0', 'native1') for i in range(count)}
    self.assertLessEqual(set(records), expected)
    for prefix in ('python', 'native0', 'native1'):
      indices = [int(record.split(':')[1]) for record in records if record.startswith(prefix + ':')]
      self.assertEqual(indices, sorted(indices))
