# flake8: noqa
import os
os.environ['FAKEUPLOAD'] = "1"

from common.apk import update_apks, start_offroad, pm_apply_packages, android_packages
from common.params import Params
from common.realtime import sec_since_boot
from common.testing import phone_only
from selfdrive.manager import manager_init, manager_prepare
from selfdrive.manager import start_managed_process, kill_managed_process, get_running
from selfdrive.manager import start_daemon_process
from functools import wraps
import json
import requests
import signal
import subprocess
import time
from datetime import datetime, timedelta

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
      if not DID_INIT:
        test_manager_prepare()

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

@phone_only
@with_processes(['loggerd', 'logmessaged', 'tombstoned', 'proclogd', 'logcatd'])
def test_logging():
  print("LOGGING IS SET UP")
  time.sleep(1.0)

@phone_only
@with_processes(['camerad', 'modeld', 'dmonitoringmodeld'])
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

@phone_only
def test_athena():
  print("ATHENA")
  start = sec_since_boot()
  start_daemon_process("manage_athenad")
  params = Params()
  manage_athenad_pid = params.get("AthenadPid")
  assert manage_athenad_pid is not None
  try:
    os.kill(int(manage_athenad_pid), 0)
    # process is running
  except OSError:
    assert False, "manage_athenad is dead"

  def expect_athena_starts(timeout=30):
    now = time.time()
    athenad_pid = None
    while athenad_pid is None:
      try:
        athenad_pid = subprocess.check_output(["pgrep", "-P", manage_athenad_pid], encoding="utf-8").strip()
        return athenad_pid
      except subprocess.CalledProcessError:
        if time.time() - now > timeout:
          assert False, f"Athena did not start within {timeout} seconds"
        time.sleep(0.5)

  def athena_post(payload, max_retries=5, wait=5):
    tries = 0
    while 1:
      try:
        resp = requests.post(
          "https://athena.comma.ai/" + params.get("DongleId", encoding="utf-8"),
          headers={
            "Authorization": "JWT " + os.getenv("COMMA_JWT"),
            "Content-Type": "application/json"
          },
          data=json.dumps(payload),
          timeout=30
        )
        resp_json = resp.json()
        if resp_json.get('error'):
          raise Exception(resp_json['error'])
        return resp_json
      except Exception as e:
        time.sleep(wait)
        tries += 1
        if tries == max_retries:
          raise
        else:
          print(f'athena_post failed {e}. retrying...')

  def expect_athena_registers(test_t0):
    resp = athena_post({
      "method": "echo",
      "params": ["hello"],
      "id": 0,
      "jsonrpc": "2.0"
    }, max_retries=12, wait=5)
    assert resp.get('result') == "hello", f'Athena failed to register ({resp})'

    last_pingtime = params.get("LastAthenaPingTime", encoding='utf8')
    assert last_pingtime, last_pingtime
    assert ((int(last_pingtime)/1e9) - test_t0) < (sec_since_boot() - test_t0)

  try:
    athenad_pid = expect_athena_starts()
    # kill athenad and ensure it is restarted (check_output will throw if it is not)
    os.kill(int(athenad_pid), signal.SIGINT)
    expect_athena_starts()

    if not os.getenv('COMMA_JWT'):
      print('WARNING: COMMA_JWT env not set, will not test requests to athena.comma.ai')
      return

    expect_athena_registers(start)

    print("ATHENA: getSimInfo")
    resp = athena_post({
      "method": "getSimInfo",
      "id": 0,
      "jsonrpc": "2.0"
    })
    assert resp.get('result'), resp
    assert 'sim_id' in resp['result'], resp['result']

    print("ATHENA: takeSnapshot")
    resp = athena_post({
      "method": "takeSnapshot",
      "id": 0,
      "jsonrpc": "2.0"
    })
    assert resp.get('result'), resp
    assert resp['result']['jpegBack'], resp['result']

    @with_processes(["thermald"])
    def test_athena_thermal():
      print("ATHENA: getMessage(thermal)")
      resp = athena_post({
        "method": "getMessage",
        "params": {"service": "thermal", "timeout": 5000},
        "id": 0,
        "jsonrpc": "2.0"
      })
      assert resp.get('result'), resp
      assert resp['result']['thermal'], resp['result']
    test_athena_thermal()
  finally:
    try:
      athenad_pid = subprocess.check_output(["pgrep", "-P", manage_athenad_pid], encoding="utf-8").strip()
    except subprocess.CalledProcessError:
      athenad_pid = None

    try:
      os.kill(int(manage_athenad_pid), signal.SIGINT)
      os.kill(int(athenad_pid), signal.SIGINT)
    except (OSError, TypeError):
      pass

# TODO: re-enable when jenkins test has /data/pythonpath -> /data/openpilot
# @phone_only
# @with_apks()
# def test_apks():
#   print("APKS")
#   time.sleep(14.0)
