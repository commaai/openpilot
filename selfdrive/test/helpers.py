import os
import time

from functools import wraps

import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.selfdrive.configs.base import Processes
from openpilot.selfdrive.manager.manager import managed_processes
from openpilot.system.hardware import PC
from openpilot.system.version import training_version, terms_version


def set_params_enabled():
  os.environ['REPLAY'] = "1"
  os.environ['FINGERPRINT'] = "TOYOTA COROLLA TSS2 2019"
  os.environ['LOGPRINT'] = "debug"

  params = Params()
  params.put("HasAcceptedTerms", terms_version)
  params.put("CompletedTrainingVersion", training_version)
  params.put_bool("OpenpilotEnabledToggle", True)

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

def with_processes(processes: Processes, init_time=0, ignore_stopped=None):
  ignore_stopped = [] if ignore_stopped is None else ignore_stopped

  def wrapper(func):
    @wraps(func)
    def wrap(*args, **kwargs):
      # start and assert started
      for n, proc in enumerate(processes):
        managed_processes[proc.name].start()
        if n < len(processes) - 1:
          time.sleep(init_time)
      assert all(managed_processes[proc.name].proc.exitcode is None for proc in processes)

      # call the function
      try:
        func(*args, **kwargs, services=processes)
        # assert processes are still started
        assert all(managed_processes[proc.name].proc.exitcode is None for proc in processes if proc.name not in ignore_stopped)
      finally:
        for proc in processes:
          managed_processes[proc.name].stop()

    return wrap
  return wrapper


def noop(*args, **kwargs):
  pass


def read_segment_list(segment_list_path):
  with open(segment_list_path, "r") as f:
    seg_list = f.read().splitlines()

  return [(platform[2:], segment) for platform, segment in zip(seg_list[::2], seg_list[1::2], strict=True)]
