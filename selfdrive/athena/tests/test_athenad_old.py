#!/usr/bin/env python3
import json
import os
import signal
import subprocess
import time

import requests

from selfdrive.manager.process_config import managed_processes
from common.params import Params
from common.realtime import sec_since_boot
from selfdrive.test.helpers import with_processes

os.environ['FAKEUPLOAD'] = "1"


def test_athena():
  print("ATHENA")
  start = sec_since_boot()
  managed_processes['manage_athenad'].start()

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
          f"https://athena.comma.ai/{params.get('DongleId', encoding='utf-8')}",
          headers={
            "Authorization": "JWT thisisnotajwt",
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

    @with_processes(["deviceStated"])
    def test_athena_deviceState():
      print("ATHENA: getMessage(deviceState)")
      resp = athena_post({
        "method": "getMessage",
        "params": {"service": "deviceState", "timeout": 5000},
        "id": 0,
        "jsonrpc": "2.0"
      })
      assert resp.get('result'), resp
      assert resp['result']['deviceState'], resp['result']
    test_athena_deviceState()
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


