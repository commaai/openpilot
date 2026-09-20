import atexit
from contextlib import ExitStack
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any
import unittest
from unittest.mock import patch

from openpilot.selfdrive.modeld.inference_process import InferenceProcess


class TestModel:
  def __init__(self, offset):
    self.offset = offset
    self.calls = 0

  def __call__(self, operation, value: Any = None):
    self.calls += 1
    if operation == "echo":
      return value
    if operation == "count":
      return self.offset + value, self.calls, os.getpid()
    if operation == "sleep":
      time.sleep(value)
      return "late result"
    if operation == "raise":
      raise ValueError("inference failed")
    if operation == "exit":
      os.kill(os.getpid(), signal.SIGKILL)
    if operation == "busy":
      signal.signal(signal.SIGTERM, signal.SIG_IGN)
      while True:
        pass
    if operation == "large_result":
      return b"x" * value
    if operation == "unserializable":
      return lambda: None
    raise ValueError(operation)


def make_model(offset=0):
  return TestModel(offset)


def finish_cleanup(done):
  time.sleep(1.1)
  done.value = True


def make_model_with_cleanup(done):
  atexit.register(finish_cleanup, done)
  return TestModel(0)


def fail_initialization():
  raise ValueError("initialization failed")


def stall_initialization():
  time.sleep(60)
  return TestModel(0)


def run_worker_parent(connection):
  with InferenceProcess(make_model, startup_timeout=2) as worker:
    connection.send(worker.process.pid)
    connection.close()
    time.sleep(60)


def linux_process_state(pid):
  try:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
  except FileNotFoundError:
    return None


class TestInferenceProcess(unittest.TestCase):
  def assert_reaped(self, worker):
    self.assertFalse(worker.process.is_alive())
    self.assertIsNotNone(worker.process.exitcode)

  def assert_no_new_children(self, previous_pids):
    self.assertEqual({child.pid for child in multiprocessing.active_children()} - previous_pids, set())

  def test_repeated_calls_preserve_worker_state_and_factory_arguments(self):
    with InferenceProcess(make_model, args=(10,)) as worker:
      pid = worker.process.pid
      self.assertNotEqual(pid, os.getpid())
      for call in range(1, 4):
        self.assertEqual(worker.call("count", call, timeout=1), (10 + call, call, pid))
    self.assert_reaped(worker)

  def test_nested_results_and_large_request(self):
    value = {"frame": 17, "inputs": b"x" * 48_000, "state": [None, True, (1.5, -2)]}
    with InferenceProcess(make_model) as worker:
      self.assertEqual(worker.call("echo", value, timeout=1), value)

  def test_large_result_does_not_block_notification(self):
    with InferenceProcess(make_model) as worker:
      self.assertEqual(worker.call("large_result", 700_000, timeout=1), b"x" * 700_000)
      self.assertEqual(worker.call("echo", "next", timeout=1), "next")

  def test_initialization_failure_reaps_child(self):
    previous_pids = {child.pid for child in multiprocessing.active_children()}
    with self.assertRaises(RuntimeError):
      InferenceProcess(fail_initialization, startup_timeout=2)
    self.assert_no_new_children(previous_pids)

  def test_interruption_after_process_start_reaps_child(self):
    previous_pids = {child.pid for child in multiprocessing.active_children()}
    start = multiprocessing.process.BaseProcess.start

    def interrupted_start(process):
      start(process)
      raise KeyboardInterrupt

    with patch.object(multiprocessing.process.BaseProcess, 'start', interrupted_start), self.assertRaises(KeyboardInterrupt):
      InferenceProcess(make_model)
    self.assert_no_new_children(previous_pids)

  def test_sigint_during_popen_reaps_child(self):
    original_handler = signal.getsignal(signal.SIGINT)
    original_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    if signal.SIGINT in original_mask:
      self.skipTest("SIGINT is blocked in the test environment")
    done = threading.Event()
    device_thread = threading.Thread(target=done.wait)
    device_thread.start()
    process_type = multiprocessing.get_context("spawn").Process
    original_popen = process_type._Popen
    created = []

    def interrupted_popen(process):
      popen = original_popen(process)
      created.append(popen)
      self.assertIsNone(process._popen)
      thread_id = device_thread.ident
      assert thread_id is not None
      signal.pthread_kill(thread_id, signal.SIGINT)
      time.sleep(0.05)
      return popen

    try:
      with patch.object(process_type, "_Popen", staticmethod(interrupted_popen)), self.assertRaises(KeyboardInterrupt):
        InferenceProcess(stall_initialization)
      self.assertEqual(len(created), 1)
      self.assertIsNotNone(created[0].poll(), "constructor lost the child before Process.start stored its handle")
      self.assertIs(signal.getsignal(signal.SIGINT), original_handler)
      self.assertEqual(signal.pthread_sigmask(signal.SIG_BLOCK, set()), original_mask)
    finally:
      for popen in created:
        if popen.poll() is None:
          popen.terminate()
          popen.wait(timeout=2)
        if popen.poll() is None:
          popen.kill()
          popen.wait(timeout=2)
        popen.close()
      done.set()
      device_thread.join(timeout=2)

  def test_cleanup_stack_owns_child_before_constructor_returns(self):
    created = []

    class InterruptedConstruction(InferenceProcess):
      def __init__(self, cleanup):
        super().__init__(make_model, cleanup=cleanup)
        created.append(self)
        os.kill(os.getpid(), signal.SIGINT)

    with self.assertRaises(KeyboardInterrupt), ExitStack() as cleanup:
      InterruptedConstruction(cleanup)
    self.assertEqual(len(created), 1)
    self.assert_reaped(created[0])
    self.assertEqual(created[0].process.exitcode, 0)

  def test_initialization_timeout_reaps_child(self):
    previous_pids = {child.pid for child in multiprocessing.active_children()}
    started = time.monotonic()
    with self.assertRaises(TimeoutError):
      InferenceProcess(stall_initialization, startup_timeout=0.1)
    self.assertLess(time.monotonic() - started, 3)
    self.assert_no_new_children(previous_pids)

  def test_first_call_timeout_returns_without_waiting_for_inference(self):
    with InferenceProcess(make_model) as worker:
      started = time.monotonic()
      with self.assertRaises(TimeoutError):
        worker.call("sleep", 60, timeout=0.03)
      self.assertLess(time.monotonic() - started, 0.5)
    self.assert_reaped(worker)

  def test_stall_after_successes_fails_channel(self):
    with InferenceProcess(make_model) as worker:
      for value in range(3):
        self.assertEqual(worker.call("echo", value, timeout=1), value)
      with self.assertRaises(TimeoutError):
        worker.call("sleep", 60, timeout=0.03)
      with self.assertRaises(RuntimeError):
        worker.call("echo", "must not reuse timed-out worker", timeout=1)
    self.assert_reaped(worker)

  def test_late_result_cannot_be_used_by_another_call(self):
    with InferenceProcess(make_model) as worker:
      with self.assertRaises(TimeoutError):
        worker.call("sleep", 0.15, timeout=0.03)
      time.sleep(0.2)
      with self.assertRaises(RuntimeError):
        worker.call("echo", "new request", timeout=1)

  def test_inference_exception_is_reported(self):
    with InferenceProcess(make_model) as worker:
      with self.assertRaises(RuntimeError):
        worker.call("raise", timeout=1)
    self.assert_reaped(worker)

  def test_unserializable_result_is_reported(self):
    with InferenceProcess(make_model) as worker:
      with self.assertRaises(RuntimeError):
        worker.call("unserializable", timeout=1)
    self.assert_reaped(worker)

  def test_worker_exit_during_call_is_reported(self):
    with InferenceProcess(make_model) as worker:
      started = time.monotonic()
      with self.assertRaises(RuntimeError):
        worker.call("exit", timeout=2)
      self.assertLess(time.monotonic() - started, 1)
    self.assert_reaped(worker)

  def test_worker_death_before_call_is_reported(self):
    with InferenceProcess(make_model) as worker:
      worker.process.terminate()
      worker.process.join(timeout=2)
      self.assert_reaped(worker)
      with self.assertRaises(RuntimeError):
        worker.call("echo", "request", timeout=1)

  def test_normal_close_allows_finalizers_to_complete(self):
    done = multiprocessing.get_context("spawn").RawValue("b", False)
    with InferenceProcess(make_model_with_cleanup, args=(done,)) as worker:
      self.assertEqual(worker.call("echo", "ready", timeout=1), "ready")
    self.assertTrue(done.value)
    self.assertEqual(worker.process.exitcode, 0)
    self.assert_reaped(worker)

  def test_close_is_idempotent_and_prevents_calls(self):
    worker = InferenceProcess(make_model)
    try:
      worker.close()
      worker.close()
      worker.stop()
      self.assert_reaped(worker)
      with self.assertRaises(RuntimeError):
        worker.call("echo", "request", timeout=1)
    finally:
      worker.close()

  def test_sigint_during_stop_preserves_reaper_and_signal_state(self):
    original_handler = signal.getsignal(signal.SIGINT)
    original_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    if signal.SIGINT in original_mask:
      self.skipTest("SIGINT is blocked in the test environment")

    def interrupt(signum, frame):
      raise KeyboardInterrupt

    for target in ("process", "device_thread"):
      with self.subTest(target=target):
        worker = InferenceProcess(make_model)
        done = threading.Event()
        device_thread = threading.Thread(target=done.wait)
        device_thread.start()
        reaper_masks = []
        send, reap = worker.connection.send_bytes, worker._reap

        def interrupted_send(message, target=target, device_thread=device_thread, send=send):
          if target == "process":
            os.kill(os.getpid(), signal.SIGINT)
          else:
            thread_id = device_thread.ident
            assert thread_id is not None
            signal.pthread_kill(thread_id, signal.SIGINT)
          send(message)

        def observed_reap(reaper_masks=reaper_masks, reap=reap):
          reaper_masks.append(signal.pthread_sigmask(signal.SIG_BLOCK, set()))
          reap()

        signal.signal(signal.SIGINT, interrupt)
        try:
          with patch.object(worker.connection, "send_bytes", interrupted_send), patch.object(worker, "_reap", observed_reap):
            with self.assertRaises(KeyboardInterrupt):
              worker.stop()
          self.assertIsNotNone(worker.reaper)
          self.assertIs(signal.getsignal(signal.SIGINT), interrupt)
          self.assertEqual(signal.pthread_sigmask(signal.SIG_BLOCK, set()), original_mask)
          worker.close()
          self.assertEqual(reaper_masks, [original_mask])
          self.assertEqual(worker.process.exitcode, 0)
          self.assert_reaped(worker)
        finally:
          signal.signal(signal.SIGINT, original_handler)
          done.set()
          device_thread.join(timeout=2)
          worker.close()

  def test_nonblocking_stop_eventually_reaps_busy_worker(self):
    worker = InferenceProcess(make_model)
    try:
      with self.assertRaises(TimeoutError):
        worker.call("busy", timeout=0.1)
      started = time.monotonic()
      worker.stop()
      self.assertLess(time.monotonic() - started, 0.2)
      deadline = time.monotonic() + 3
      while worker.process.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
      self.assert_reaped(worker)
    finally:
      worker.close()

  def test_busy_worker_close_is_bounded_and_replacement_works(self):
    worker = InferenceProcess(make_model)
    try:
      with self.assertRaises(TimeoutError):
        worker.call("busy", timeout=0.1)
      started = time.monotonic()
      worker.close()
      self.assertLess(time.monotonic() - started, 3)
      self.assert_reaped(worker)
    finally:
      worker.close()
    with InferenceProcess(make_model, args=(100,)) as replacement:
      self.assertNotEqual(replacement.process.pid, worker.process.pid)
      self.assertEqual(replacement.call("count", 5, timeout=1)[:2], (105, 1))
    self.assert_reaped(replacement)

  @unittest.skipUnless(sys.platform == "linux", "parent-death signal is Linux-specific")
  def test_abrupt_parent_death_stops_worker(self):
    ctx = multiprocessing.get_context("spawn")
    receiver, sender = ctx.Pipe(duplex=False)
    parent = ctx.Process(target=run_worker_parent, args=(sender,))
    worker_pid = None
    parent.start()
    sender.close()
    try:
      self.assertTrue(receiver.poll(5), "worker parent failed to report startup")
      worker_pid = receiver.recv()
      self.assertNotIn(linux_process_state(worker_pid), (None, "Z"))
      parent.kill()
      parent.join(timeout=2)
      self.assertFalse(parent.is_alive())
      deadline = time.monotonic() + 3
      while linux_process_state(worker_pid) not in (None, "Z") and time.monotonic() < deadline:
        time.sleep(0.01)
      # An orphan zombie has exited and released its resources; its reaping is
      # owned by init, whose timing is outside this worker's control.
      self.assertTrue(linux_process_state(worker_pid) in (None, "Z"))
    finally:
      receiver.close()
      if parent.is_alive():
        parent.kill()
      parent.join(timeout=2)
      if worker_pid is not None and linux_process_state(worker_pid) not in (None, "Z"):
        try:
          os.kill(worker_pid, signal.SIGKILL)
        except ProcessLookupError:
          pass


if __name__ == "__main__":
  unittest.main()
