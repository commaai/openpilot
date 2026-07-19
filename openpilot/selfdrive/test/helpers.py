import contextlib
import gc
import http.server
import os
import subprocess
import threading
import time
import unittest

from functools import wraps

import openpilot.cereal.messaging as messaging
from openpilot.common.hardware import TICI, HARDWARE
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.system.manager.process_config import managed_processes
from openpilot.common.version import training_version, terms_version

# tests that take awhile; skipped unless RUN_SLOW is set (run_tests.py sets it when a file is explicitly named)
slow = unittest.skipUnless(os.getenv("RUN_SLOW"), "slow test, set RUN_SLOW=1 to run")


class OpenpilotTestCase(unittest.TestCase):
  """Every openpilot test runs with a clean env and a fresh OpenpilotPrefix per test method.

     Subclasses overriding setUp/setUpClass MUST call super() first."""

  SHARED_DOWNLOAD_CACHE = False
  SKIP_TICI_SETUP = False

  @classmethod
  def setUpClass(cls):
    env = dict(os.environ)
    cls.addClassCleanup(lambda: (os.environ.clear(), os.environ.update(env)))

  def setUp(self):
    # cleanups run LIFO: prefix-unchanged assert -> prefix exit -> manager/gc cleanup -> env restore
    env = dict(os.environ)
    self.addCleanup(lambda: (os.environ.clear(), os.environ.update(env)))
    self.addCleanup(self._post_prefix_cleanup)
    self._op_prefix = self.enterContext(OpenpilotPrefix(shared_download_cache=self.SHARED_DOWNLOAD_CACHE))
    self.addCleanup(self._assert_prefix_unchanged)

    if TICI and not self.SKIP_TICI_SETUP:
      # ensure a consistent state on-device
      HARDWARE.initialize_hardware()
      HARDWARE.set_power_save(False)
      subprocess.call(["pkill", "-9", "-f", "athena"])

  def _assert_prefix_unchanged(self):
    assert os.environ.get("OPENPILOT_PREFIX") == self._op_prefix.prefix, "test changed OPENPILOT_PREFIX"

  @staticmethod
  def _post_prefix_cleanup():
    from openpilot.system.manager import manager  # lazy: helpers is imported by the sim bridge at runtime
    manager.manager_cleanup()

    # some processes disable gc for performance, re-enable here
    if not gc.isenabled():
      gc.enable()
      gc.collect()


def set_params_enabled():
  os.environ['FINGERPRINT'] = "TOYOTA_COROLLA_TSS2"
  os.environ['LOGPRINT'] = "debug"

  params = Params()
  params.put("HasAcceptedTerms", terms_version, block=True)
  params.put("CompletedTrainingVersion", training_version, block=True)
  params.put_bool("OpenpilotEnabledToggle", True, block=True)

  # valid calib
  msg = messaging.new_message('liveCalibration')
  msg.liveCalibration.validBlocks = 20
  msg.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
  params.put("CalibrationParams", msg.to_bytes(), block=True)

def release_only(f):
  @wraps(f)
  def wrap(self, *args, **kwargs):
    if "RELEASE" not in os.environ:
      raise unittest.SkipTest("This test is only for release branches")
    f(self, *args, **kwargs)
  return wrap


def collect_logs(services, duration):
  socks = [messaging.sub_sock(s, conflate=False, timeout=100) for s in services]
  logs = []
  start = time.monotonic()
  while time.monotonic() - start < duration:
    for s in socks:
      logs.extend(messaging.drain_sock(s))
  return logs


@contextlib.contextmanager
def log_collector(services):
  """Background thread that continuously drains messages from services.
     Use when the main thread needs to do blocking work (e.g. capturing images)."""
  socks = [messaging.sub_sock(s, conflate=False, timeout=100) for s in services]
  raw_logs = []
  lock = threading.Lock()
  stop_event = threading.Event()

  def _drain():
    while not stop_event.is_set():
      for s in socks:
        msgs = messaging.drain_sock(s)
        if msgs:
          with lock:
            raw_logs.extend(msgs)
      time.sleep(0.01)

  thread = threading.Thread(target=_drain, daemon=True)
  thread.start()
  try:
    yield raw_logs, lock
  finally:
    stop_event.set()
    thread.join(timeout=2)


@contextlib.contextmanager
def processes_context(processes, init_time=0, ignore_stopped=None):
  ignore_stopped = [] if ignore_stopped is None else ignore_stopped

  # start and assert started
  for n, p in enumerate(processes):
    managed_processes[p].start()
    if n < len(processes) - 1:
      time.sleep(init_time)

  assert all(managed_processes[name].proc.exitcode is None for name in processes)

  try:
    yield [managed_processes[name] for name in processes]
    # assert processes are still started
    assert all(managed_processes[name].proc.exitcode is None for name in processes if name not in ignore_stopped)
  finally:
    for p in processes:
      managed_processes[p].stop()


def with_processes(processes, init_time=0, ignore_stopped=None):
  def wrapper(func):
    @wraps(func)
    def wrap(*args, **kwargs):
      with processes_context(processes, init_time, ignore_stopped):
        return func(*args, **kwargs)

    return wrap
  return wrapper


def noop(*args, **kwargs):
  pass


def read_segment_list(segment_list_path):
  with open(segment_list_path) as f:
    seg_list = f.read().splitlines()

  return [(platform[2:], segment) for platform, segment in zip(seg_list[::2], seg_list[1::2], strict=True)]


@contextlib.contextmanager
def http_server_context(handler, setup=None):
  host = '127.0.0.1'
  server = http.server.HTTPServer((host, 0), handler)
  port = server.server_port
  t = threading.Thread(target=server.serve_forever)
  t.start()

  if setup is not None:
    setup(host, port)

  try:
    yield (host, port)
  finally:
    server.shutdown()
    server.server_close()
    t.join()


def with_http_server(func, handler=http.server.BaseHTTPRequestHandler, setup=None):
  @wraps(func)
  def inner(*args, **kwargs):
    with http_server_context(handler, setup) as (host, port):
      return func(*args, f"http://{host}:{port}", **kwargs)
  return inner


def DirectoryHttpServer(directory) -> type[http.server.SimpleHTTPRequestHandler]:
  # creates an http server that serves files from directory
  class Handler(http.server.SimpleHTTPRequestHandler):
    API_NO_RESPONSE = False
    API_BAD_RESPONSE = False

    def do_GET(self):
      if self.API_NO_RESPONSE:
        return

      if self.API_BAD_RESPONSE:
        self.send_response(500, "")
        return
      super().do_GET()

    def __init__(self, *args, **kwargs):
      super().__init__(*args, directory=str(directory), **kwargs)

  return Handler
