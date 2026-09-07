from concurrent.futures import ThreadPoolExecutor
import glob
import json
import multiprocessing
from pathlib import Path
import signal
import shutil
import subprocess
import threading
from unittest import mock
import os
import time

from openpilot.common.test import OpenpilotTestCase
import openpilot.cereal.messaging as messaging
from openpilot.system.manager.process_config import managed_processes
from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog, ipchandler
from openpilot.common.file_queue import FileQueue


SWAGLOG_TEST = Path(__file__).resolve().parents[2] / "common/tests/test_swaglog"


def produce_logs(prefix, count):
  for i in range(count):
    cloudlog.debug(f'{prefix}:{i}')


def crash_during_log(connection):
  write = os.write

  def partial_write(fd, data):
    write(fd, data[:16])
    connection.send(True)
    signal.pause()  # Parent kills us with an unpublished, partially written file.
    return 16

  with mock.patch('os.write', partial_write):
    cloudlog.debug('incomplete')


def produce_burst(start, count, size):
  start.wait(10)
  produce_logs('a' * size, count)


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

  def _wait_for(self, condition):
    deadline = time.monotonic() + 10
    while not condition():
      self.assertLess(time.monotonic(), deadline, "timed out waiting for logmessaged")
      time.sleep(0.01)

  def test_simple_log(self):
    managed_processes['logmessaged'].start()
    ready = set()

    def connected():
      cloudlog.error('logmessaged readiness probe')
      for name, sock in (('log', self.sock), ('error', self.error_sock)):
        if messaging.drain_sock(sock):
          ready.add(name)
      return len(ready) == 2

    self._wait_for(connected)
    msgs = [f"abc {i}" for i in range(10)]
    for m in msgs:
      cloudlog.error(m)
    received, errors = [], []

    def collected():
      received.extend(msg for m in messaging.drain_sock(self.sock) if (msg := json.loads(m.logMessage)['msg']) in msgs)
      errors.extend(msg for m in messaging.drain_sock(self.error_sock) if (msg := json.loads(m.errorLogMessage)['msg']) in msgs)
      return len(received) >= len(msgs) and len(errors) >= len(msgs)

    self._wait_for(collected)
    self.assertEqual(received, msgs)
    self.assertEqual(errors, msgs)
    assert len(self._get_log_files()) >= 1

  def test_big_log(self):
    n = 10
    msg = "a"*3*1024*1024
    for _ in range(n):
      cloudlog.info(msg)
    # The complete burst must fit while the daemon is stopped.
    managed_processes['logmessaged'].start()
    self._wait_for(lambda: sum(os.path.getsize(f) for f in self._get_log_files()) > n*len(msg))

    msgs = messaging.drain_sock(self.sock)
    assert len(msgs) == 0

    logsize = sum([os.path.getsize(f) for f in self._get_log_files()])
    assert (n*len(msg)) < logsize < (n*(len(msg)+1024))

  def test_swaglog_daemon_restart(self):
    producer = ipchandler.queue

    def logged(message):
      return any(message in Path(f).read_text() for f in self._get_log_files())

    cloudlog.info('before daemon restart')
    managed_processes['logmessaged'].start()
    self._wait_for(lambda: logged('before daemon restart'))
    managed_processes['logmessaged'].stop(block=True)
    cloudlog.info('during daemon restart')
    managed_processes['logmessaged'].start()
    self._wait_for(lambda: logged('during daemon restart'))

    cloudlog.info('after daemon restart')
    self._wait_for(lambda: logged('after daemon restart'))
    self.assertIs(ipchandler.queue, producer)

  def _emit(self, message, native):
    if native:
      subprocess.run([str(SWAGLOG_TEST), '--emit'], input=message.encode(), check=True, timeout=10)
    else:
      cloudlog.debug(message)

  def _reader(self):
    return FileQueue(Paths.swaglog_ipc())

  def _read_log(self, reader):
    data = reader.receive()
    self.assertIsNotNone(data)
    self.assertEqual(data[0], 10)
    record = json.loads(data[1:])
    self.assertEqual(record['levelnum'], 10)
    return record['msg']

  def test_swaglog_reopen(self):
    for native in (False, True):
      with self.subTest(native=native):
        message = 'a' * (3 * 1024 * 1024)
        self._emit(message, native)
        reopened = self._reader()
        self.assertEqual(self._read_log(reopened), message)
        self.assertIsNone(reopened.receive())
    ipchandler.close()
    self._emit('after close', False)
    self.assertEqual(self._read_log(self._reader()), 'after close')

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

  def test_swaglog_crash_and_stale_cleanup(self):
    reader = self._reader()
    cloudlog.debug('committed')
    pending = Path(Paths.swaglog_ipc()) / 'pending'
    ctx = multiprocessing.get_context('fork')
    parent, child = ctx.Pipe()
    self.addCleanup(parent.close)
    self.addCleanup(child.close)
    process = ctx.Process(target=crash_during_log, args=(child,))
    process.start()
    try:
      self.assertTrue(parent.poll(10))
      self.assertTrue(parent.recv())
      self.assertTrue(list(pending.iterdir()))
      self.assertEqual(self._read_log(reader), 'committed')
      self.assertIsNone(reader.receive())
      self.assertTrue(list(pending.iterdir()), 'active producer reservation was deleted')
      for native in (False, True):
        self._emit('while another producer is stalled', native)
        self.assertEqual(self._read_log(reader), 'while another producer is stalled')
    finally:
      process.kill()
      process.join(10)
    self.assertIsNone(reader.receive())
    self.assertFalse(list(pending.iterdir()), 'dead producer reservation was not cleaned')
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

  def test_swaglog_fork_with_new_prefix(self):
    original = self._reader()
    cloudlog.debug('parent prefix')
    self.assertEqual(self._read_log(original), 'parent prefix')
    child_prefix = os.environ['OPENPILOT_PREFIX'] + '_child'
    with mock.patch.dict(os.environ, {'OPENPILOT_PREFIX': child_prefix}):
      child_path = Paths.swaglog_ipc()
      self.addCleanup(shutil.rmtree, child_path, True)
      child_reader = FileQueue(child_path)
      process = multiprocessing.get_context('fork').Process(target=produce_logs, args=('child prefix', 1))
      process.start()
      try:
        process.join(10)
        self.assertEqual(process.exitcode, 0)
      finally:
        if process.is_alive():
          process.kill()
          process.join()
      self.assertEqual(self._read_log(child_reader), 'child prefix:0')
      self.assertIsNone(child_reader.receive())
    self.assertIsNone(original.receive())
    cloudlog.debug('parent still connected')
    self.assertEqual(self._read_log(original), 'parent still connected')

  def test_swaglog_concurrent_producers(self):
    reader = self._reader()
    count = 200
    done = threading.Event()

    def consume():
      records = []
      while True:
        data = reader.receive()
        if data is not None:
          self.assertEqual(data[0], 10)
          records.append(json.loads(data[1:])['msg'])
        elif done.is_set():
          return records
        else:
          done.wait(0.001)

    processes = []
    with ThreadPoolExecutor(max_workers=1) as executor:
      consumer = executor.submit(consume)
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
        done.set()
      records = consumer.result(timeout=10)
    # These small messages fit comfortably below the cap, so none may be dropped.
    self.assertEqual(len(records), len(set(records)))
    expected = {f'{prefix}:{i}' for prefix in ('python', 'native0', 'native1') for i in range(count)}
    self.assertEqual(set(records), expected)
    for prefix in ('python', 'native0', 'native1'):
      indices = [int(record.split(':')[1]) for record in records if record.startswith(prefix + ':')]
      self.assertTrue(indices, f'no logs from {prefix}')
      self.assertEqual(indices, sorted(indices))

  def test_swaglog_burst_capacity(self):
    capacity = FileQueue.CAPACITY
    message = 'a' * (8 * 1024 * 1024)
    ctx = multiprocessing.get_context('fork')
    start = ctx.Event()
    done = threading.Event()
    root = Path(Paths.swaglog_ipc())
    page_size = os.sysconf('SC_PAGE_SIZE')

    def spool_size():
      # A rename can move a file between scans; count each reservation name once.
      sizes = {}
      for directory in ('pending', 'ready'):
        for path in (root / directory).iterdir():
          try:
            size = path.stat().st_size
            sizes[path.name] = page_size + ((size + page_size - 1) // page_size) * page_size
          except FileNotFoundError:
            pass
      return sum(sizes.values())

    def monitor():
      peak = 0
      while not done.is_set():
        peak = max(peak, spool_size())
        done.wait(0.001)
      return max(peak, spool_size())

    python = [ctx.Process(target=produce_burst, args=(start, 4, len(message))) for _ in range(4)]
    native = []
    with ThreadPoolExecutor(max_workers=1) as executor:
      usage = executor.submit(monitor)
      try:
        for process in python:
          process.start()
        for _ in range(4):
          process = subprocess.Popen([str(SWAGLOG_TEST), '--emit', '4'], stdin=subprocess.PIPE)
          native.append(process)
          process.stdin.write(message.encode())
        # Native emitters wait for EOF, Python emitters wait for the event.
        start.set()
        for process in native:
          process.stdin.close()
        for process in python:
          process.join(30)
          self.assertEqual(process.exitcode, 0)
        for process in native:
          self.assertEqual(process.wait(timeout=30), 0)
      finally:
        for process in python:
          if process.pid is not None and process.is_alive():
            process.kill()
            process.join()
        for process in native:
          if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
          if process.poll() is None:
            process.kill()
            process.wait()
        done.set()
      self.assertLessEqual(usage.result(timeout=10), 2 * capacity)
    self.assertGreater(spool_size(), capacity // 2)
    self.assertFalse(list((root / 'pending').iterdir()))
    reader = self._reader()
    while (data := reader.receive()) is not None:
      self.assertEqual(data[0], 10)
      payload, index = json.loads(data[1:])['msg'].rsplit(':', 1)
      self.assertEqual(payload, message)
      self.assertIn(int(index), range(4))
    self.assertEqual(spool_size(), 0)
