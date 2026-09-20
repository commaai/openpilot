import atexit
from contextlib import ExitStack
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, cast
import unittest
from unittest.mock import patch

from openpilot.common.parameterized import parameterized
from openpilot.selfdrive.modeld import inference_process
from openpilot.selfdrive.modeld.inference_process import InferenceProcess


def finish_cleanup(done):
  time.sleep(1.1)
  done.value = True


def partial_result(buffer, value):
  buffer[8:100008] = b'x' * 100000
  time.sleep(60)


class Model:
  def __init__(self, delay=0., fail=False, done=None, partial=False):
    time.sleep(delay)
    if fail:
      raise ValueError('initialization failed')
    if done is not None:
      atexit.register(finish_cleanup, done)
    if partial:
      patch.object(inference_process, '_write', partial_result).start()
    self.calls = 0

  def __call__(self, operation, value: Any = None):
    self.calls += 1
    if operation == 'sleep':
      time.sleep(value)
    elif operation == 'raise':
      raise ValueError('inference failed')
    elif operation == 'exit':
      os.kill(os.getpid(), signal.SIGKILL)
    elif operation == 'busy':
      signal.signal(signal.SIGTERM, signal.SIG_IGN)
      while True:
        pass
    elif operation == 'unserializable':
      return lambda: None
    elif operation == 'large':
      value = b'x' * value
    return value, self.calls, os.getpid()


def worker_parent(connection):
  worker = InferenceProcess(Model)
  connection.send(worker.process.pid)
  connection.close()
  time.sleep(60)


def process_state(pid):
  try:
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
  except FileNotFoundError:
    return None


def cleanup_popen(popen):
  if popen.poll() is None:
    popen.kill()
    popen.wait(timeout=2)
  popen.close()


class TestInferenceProcess(unittest.TestCase):
  def setUp(self):
    children = {child.pid for child in mp.active_children()}
    self.addCleanup(lambda: self.assertEqual({child.pid for child in mp.active_children()}, children))

  def worker(self, *args):
    worker = InferenceProcess(Model, args=args)
    self.addCleanup(worker.close)
    return worker

  def test_state_payloads_and_repeated_lifecycle(self):
    for _ in range(2):
      done = mp.get_context('spawn').RawValue('b', False)
      worker = self.worker(0., False, done)
      self.assertNotEqual(worker.process.pid, os.getpid())
      nested = {'data': b'x' * 48000, 'state': [None, True, (1.5, -2)]}
      for count, (operation, value) in enumerate((('echo', nested), ('large', 700000), ('echo', 'next')), 1):
        expected = b'x' * cast(int, value) if operation == 'large' else value
        self.assertEqual(worker.call(operation, value, timeout=1), (expected, count, worker.process.pid))
      worker.close()
      self.assertTrue(done.value)
      self.assertEqual(worker.process.exitcode, 0)
      with self.assertRaises(RuntimeError):
        worker.call('echo', 'closed')

  @parameterized.expand([(0., True, RuntimeError), (60., False, TimeoutError)])
  def test_startup_failures(self, delay, fail, error):
    started = time.monotonic()
    with self.assertRaises(error):
      InferenceProcess(Model, args=(delay, fail), startup_timeout=.1 if delay else 2)
    self.assertLess(time.monotonic() - started, 3.)

  @parameterized.expand([('raise', None, 0), ('unserializable', None, 0), ('exit', None, 0), ('dead', None, 0), ('busy', None, 0),
                         ('sleep', 60, 0), ('sleep', 60, 3), ('sleep', .15, 0), ('partial', None, 0)])
  def test_failed_request_is_bounded_and_cannot_be_reused(self, operation, value, successes):
    worker = self.worker(0., False, None, operation == 'partial')
    if operation == 'dead':
      worker.process.terminate()
      worker.process.join(timeout=2)
    for _ in range(successes):
      worker.call('echo', timeout=1)
    started = time.monotonic()
    error = TimeoutError if operation in ('sleep', 'busy', 'partial') else (RuntimeError, EOFError, OSError)
    timeout = .03 if error is TimeoutError else 1.
    with self.assertRaises(error):
      worker.call(operation, value, timeout=timeout)
    self.assertLess(time.monotonic() - started, max(.5, timeout))
    if operation == 'sleep' and value == .15:
      time.sleep(.2)  # Allow a late completion to arrive before trying to reuse the channel.
    with self.assertRaises(RuntimeError):
      worker.call('echo', 'must not reuse')
    started = time.monotonic()
    worker.reaper.join(3)
    self.assertIsNotNone(worker.process.exitcode)
    worker.close()
    self.assertLess(time.monotonic() - started, 3.)

  @parameterized.expand([(stage, thread) for stage in ('popen', 'registered', 'stop') for thread in (False, True)])
  def test_sigint_preserves_ownership_and_signal_state(self, stage, from_thread):
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    if signal.SIGINT in mask:
      self.skipTest('SIGINT is blocked')
    handler = signal.getsignal(signal.SIGINT)
    self.addCleanup(signal.signal, signal.SIGINT, handler)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    done = threading.Event()
    thread = threading.Thread(target=done.wait)
    thread.start()
    self.addCleanup(thread.join, 2)
    self.addCleanup(done.set)
    handles = []

    def interrupt():
      assert thread.ident is not None
      if from_thread:
        signal.pthread_kill(thread.ident, signal.SIGINT)
      else:
        os.kill(os.getpid(), signal.SIGINT)
      time.sleep(.01)

    if stage == 'popen':
      process_type = mp.get_context('spawn').Process
      original = process_type._Popen

      def popen(process):
        handle = original(process)
        handles.append(handle)
        self.addCleanup(cleanup_popen, handle)
        self.assertIsNone(process._popen)
        interrupt()
        return handle

      with patch.object(process_type, '_Popen', staticmethod(popen)), self.assertRaises(KeyboardInterrupt):
        InferenceProcess(Model)
    elif stage == 'registered':
      with ExitStack() as cleanup, self.assertRaises(KeyboardInterrupt):
        worker = InferenceProcess(Model, cleanup=cleanup)
        handles.append(worker.process._popen)
        interrupt()  # Caller has not yet installed its own cleanup.
      self.assertEqual(worker.process.exitcode, 0)
    else:
      worker = self.worker()
      handles.append(worker.process._popen)
      original, reap = worker.connection.send_bytes, worker._reap
      reaper_masks = []

      def send(message):
        interrupt()
        original(message)

      def observe_reap():
        reaper_masks.append(signal.pthread_sigmask(signal.SIG_BLOCK, set()))
        reap()

      with patch.object(worker.connection, 'send_bytes', send), patch.object(worker, '_reap', observe_reap), self.assertRaises(KeyboardInterrupt):
        worker.stop()
      worker.close()
      self.assertEqual(worker.process.exitcode, 0)
      self.assertEqual(reaper_masks, [mask])
    self.assertTrue(all(handle.poll() is not None for handle in handles))
    self.assertIs(signal.getsignal(signal.SIGINT), signal.default_int_handler)
    self.assertEqual(signal.pthread_sigmask(signal.SIG_BLOCK, set()), mask)

  @unittest.skipUnless(sys.platform == 'linux', 'parent-death signal is Linux-specific')
  def test_parent_death_stops_worker(self):
    ctx = mp.get_context('spawn')
    receiver, sender = ctx.Pipe(duplex=False)
    parent = ctx.Process(target=worker_parent, args=(sender,))
    worker_pid = None
    parent.start()
    sender.close()
    try:
      self.assertTrue(receiver.poll(5))
      worker_pid = receiver.recv()
      self.assertNotIn(process_state(worker_pid), (None, 'Z'))
      parent.kill()
      parent.join(timeout=2)
      self.assertFalse(parent.is_alive())
      deadline = time.monotonic() + 3
      while process_state(worker_pid) not in (None, 'Z') and time.monotonic() < deadline:
        time.sleep(.01)
      self.assertTrue(process_state(worker_pid) in (None, 'Z'))  # init owns reaping an exited orphan.
    finally:
      receiver.close()
      if parent.is_alive():
        parent.kill()
      parent.join(timeout=2)
      if worker_pid is not None and process_state(worker_pid) not in (None, 'Z'):
        try:
          os.kill(worker_pid, signal.SIGKILL)
        except ProcessLookupError:
          pass
