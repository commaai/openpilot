import json
import os
import requests
import signal
import subprocess
import time
import unittest

os.environ['FAKEUPLOAD'] = "1"

from common.params import Params
from common.realtime import sec_since_boot
import selfdrive.manager as manager
from selfdrive.hardware import EON
from selfdrive.test.helpers import with_processes

# TODO: make eon fast
MAX_STARTUP_TIME = 30 if EON else 15
ALL_PROCESSES = manager.persistent_processes + manager.car_started_processes

class TestManager(unittest.TestCase):

  def setUp(self):
    os.environ['PASSIVE'] = '0'

  def tearDown(self):
    manager.cleanup_all_processes(None, None)

  def test_manager_prepare(self):
    os.environ['PREPAREONLY'] = '1'
    manager.main()

  def test_startup_time(self):
    for _ in range(10):
      start = time.monotonic()
      os.environ['PREPAREONLY'] = '1'
      manager.main()
      t = time.monotonic() - start
      assert t < MAX_STARTUP_TIME, f"startup took {t}s, expected <{MAX_STARTUP_TIME}s"

  # ensure all processes exit cleanly
  def test_clean_exit(self):
    manager.manager_prepare()
    for p in ALL_PROCESSES:
      manager.start_managed_process(p)
    
    time.sleep(10)

    for p in reversed(ALL_PROCESSES):
      exit_code = manager.kill_managed_process(p, retry=False)
      if not EON and (p == 'ui'or p == 'loggerd'):
        # TODO: make Qt UI exit gracefully and fix OMX encoder exiting
        continue

      # TODO: interrupted blocking read exits with 1 in cereal. use a more unique return code
      exit_codes = [0, 1]
      if p in manager.kill_processes:
        exit_codes = [-signal.SIGKILL]
      assert exit_code in exit_codes, f"{p} died with {exit_code}"


def test_athena():
  print("ATHENA")
  start = sec_since_boot()
  manager.start_daemon_process("manage_athenad")
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


if __name__ == "__main__":
  unittest.main()
