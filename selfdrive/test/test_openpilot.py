import os
os.environ['FAKEUPLOAD'] = "1"

from common.testing import phone_only
from selfdrive.manager import manager_init, manager_prepare
from selfdrive.manager import start_managed_process, kill_managed_process, get_running
from functools import wraps
import time

DID_INIT = False

# must run first
@phone_only
def test_manager_prepare():
  global DID_INIT
  manager_init()
  manager_prepare()
  DID_INIT = True

def with_processes(processes):
  def wrapper(func):
    @wraps(func)
    def wrap():
      if not DID_INIT:
        test_manager_prepare()

      # start and assert started
      [start_managed_process(p) for p in processes]
      assert all(get_running()[name].exitcode is None for name in processes)

      # call the function
      func()

      # assert processes are still started
      assert all(get_running()[name].exitcode is None for name in processes)

      # kill and assert all stopped
      [kill_managed_process(p) for p in processes]
      assert len(get_running()) == 0
    return wrap
  return wrapper

#@phone_only
#@with_processes(['controlsd', 'radard'])
#def test_controls():
#  from selfdrive.test.longitudinal_maneuvers.plant import Plant
#
#  # start the fake car for 2 seconds
#  plant = Plant(100)
#  for i in range(200):
#    if plant.rk.frame >= 20 and plant.rk.frame <= 25:
#      cruise_buttons = CruiseButtons.RES_ACCEL
#      # rolling forward
#      assert plant.speed > 0
#    else:
#      cruise_buttons = 0
#    plant.step(cruise_buttons = cruise_buttons)
#  plant.close()
#
#  # assert that we stopped
#  assert plant.speed == 0.0

@phone_only
@with_processes(['loggerd', 'logmessaged', 'tombstoned', 'proclogd', 'logcatd'])
def test_logging():
  print("LOGGING IS SET UP")
  time.sleep(1.0)

@phone_only
@with_processes(['visiond'])
def test_visiond():
  print("VISIOND IS SET UP")
  time.sleep(5.0)

@phone_only
@with_processes(['sensord'])
def test_sensord():
  print("SENSORS ARE SET UP")
  time.sleep(1.0)

@phone_only
@with_processes(['ui'])
def test_ui():
  print("RUNNING UI")
  time.sleep(1.0)

# will have one thing to upload if loggerd ran
# TODO: assert it actually uploaded
@phone_only
@with_processes(['uploader'])
def test_uploader():
  print("UPLOADER")
  time.sleep(10.0)
