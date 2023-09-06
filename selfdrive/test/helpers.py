import os
import time
import tempfile
from typing import List, Union

from unittest import mock
from functools import wraps

import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.selfdrive.manager.process_config import managed_processes
from openpilot.system.hardware import PC
from openpilot.system.version import training_version, terms_version

SKIP_ENV_VAR = "SKIP_LONG_TESTS"

def set_params_enabled():
  os.environ['PASSIVE'] = "0"
  os.environ['REPLAY'] = "1"
  os.environ['FINGERPRINT'] = "TOYOTA COROLLA TSS2 2019"
  os.environ['LOGPRINT'] = "debug"

  params = Params()
  params.put("HasAcceptedTerms", terms_version)
  params.put("CompletedTrainingVersion", training_version)
  params.put_bool("OpenpilotEnabledToggle", True)
  params.put_bool("Passive", False)

  # valid calib
  msg = messaging.new_message('liveCalibration')
  msg.liveCalibration.validBlocks = 20
  msg.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
  params.put("CalibrationParams", msg.to_bytes())

def phone_only(f):
  @wraps(f)
  def wrap(self, *args, **kwargs):
    if PC:
      self.skipTest("This test is not meant to run on PC")
    f(self, *args, **kwargs)
  return wrap

def release_only(f):
  @wraps(f)
  def wrap(self, *args, **kwargs):
    if "RELEASE" not in os.environ:
      self.skipTest("This test is only for release branches")
    f(self, *args, **kwargs)
  return wrap

def with_processes(processes, init_time=0, ignore_stopped=None):
  ignore_stopped = [] if ignore_stopped is None else ignore_stopped

  def wrapper(func):
    @wraps(func)
    def wrap(*args, **kwargs):
      # start and assert started
      for n, p in enumerate(processes):
        managed_processes[p].start()
        if n < len(processes) - 1:
          time.sleep(init_time)
      assert all(managed_processes[name].proc.exitcode is None for name in processes)

      # call the function
      try:
        func(*args, **kwargs)
        # assert processes are still started
        assert all(managed_processes[name].proc.exitcode is None for name in processes if name not in ignore_stopped)
      finally:
        for p in processes:
          managed_processes[p].stop()

    return wrap
  return wrapper


def temporary_mock_dir(mock_paths_in: Union[List[str], str], kwarg: Union[str, None] = None, generator = tempfile.TemporaryDirectory):
  """
  mock_paths_in: string or string list representing the full path of the variable you want to mock.
  kwarg: str or None representing the kwarg that gets passed into the test function, in case the test needs access to the temporary directory.
  generator: a context to use to generate the temporary directory
  """
  def wrapper(func):
    @wraps(func)
    def wrap(*args, **kwargs):
      mock_paths = mock_paths_in if isinstance(mock_paths_in, list) else [mock_paths_in]
      with generator() as temp_dir:
        mocks = []
        for mock_path in mock_paths:
          mocks.append(mock.patch(mock_path, str(temp_dir)))
        [mock.start() for mock in mocks]
        try:
          if kwarg is not None:
            kwargs[kwarg] = temp_dir
          func(*args, **kwargs)
        finally:
          [mock.stop() for mock in mocks]

    return wrap
  return wrapper

def string_context(context):
  class StringContext:
    def __enter__(self):
      return context

    def __exit__(self, *args):
      pass

  return StringContext

temporary_dir = temporary_mock_dir([], "temp_dir")
temporary_cache_dir = temporary_mock_dir("openpilot.tools.lib.url_file.CACHE_DIR")
temporary_swaglog_dir = temporary_mock_dir("openpilot.system.swaglog.SWAGLOG_DIR", "temp_dir")
temporary_laikad_downloads_dir = temporary_mock_dir("openpilot.selfdrive.locationd.laikad.DOWNLOADS_CACHE_FOLDER")
temporary_swaglog_ipc = temporary_mock_dir(["openpilot.system.swaglog.SWAGLOG_IPC", "system.logmessaged.SWAGLOG_IPC"],
                                           generator=string_context("/tmp/test_swaglog_ipc"))