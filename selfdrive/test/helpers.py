import subprocess
from functools import wraps
from nose.tools import nottest

from common.android import ANDROID
from common.apk import update_apks, start_offroad, pm_apply_packages, android_packages
from selfdrive.manager import start_managed_process, kill_managed_process, get_running

def phone_only(x):
  if ANDROID:
    return x
  else:
    return nottest(x)

def with_processes(processes):
  def wrapper(func):
    @wraps(func)
    def wrap():
      # start and assert started
      [start_managed_process(p) for p in processes]
      assert all(get_running()[name].exitcode is None for name in processes)

      # call the function
      try:
        func()
        # assert processes are still started
        assert all(get_running()[name].exitcode is None for name in processes)
      finally:
        # kill and assert all stopped
        [kill_managed_process(p) for p in processes]
        assert len(get_running()) == 0
    return wrap
  return wrapper

def with_apks():
  def wrapper(func):
    @wraps(func)
    def wrap():
      update_apks()
      pm_apply_packages('enable')
      start_offroad()

      func()

      try:
        for package in android_packages:
          apk_is_running = (subprocess.call(["pidof", package]) == 0)
          assert apk_is_running, package
      finally:
        pm_apply_packages('disable')
        for package in android_packages:
          apk_is_not_running = (subprocess.call(["pidof", package]) == 1)
          assert apk_is_not_running, package
    return wrap
  return wrapper

