#!/usr/bin/env python3
# every test route's engagement on a DUT in an ultimate provisioning (UP) station, the UP plays the route's CAN raw.
# ALFRED_SHARD/ALFRED_SHARDS drive every Nth route, SCENARIOS=TOYOTA_COROLLA_TSS2,HONDA_CIVIC only those car models or routes
import os
import signal
import subprocess
import sys
import time

import openpilot.cereal.messaging as messaging
from opendbc.car.tests.routes import CarTestRoute, routes
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.selfdrive.test.cars.route import DT, LEAD, Window, find_window
from openpilot.selfdrive.test.cars.rules import check
from openpilot.selfdrive.test.helpers import set_params_enabled

PANDA_LIMIT = 45.
ROAD_CAMERAS = ('narrowRoadCameraState', 'wideRoadCameraState')
RECORD = ('can', 'pandaStates', 'selfdriveState', 'carState', 'carParams', 'cabinCameraState', *ROAD_CAMERAS)


class RigProblem(Exception):
  pass


def up_call(up, call: str, *args):
  # the UP not answering is the rack's problem
  try:
    return getattr(up, call)(*args)
  except Exception as e:
    raise RigProblem(f"rig problem: UP {call} failed: {e!r}") from e


def start_manager(route: CarTestRoute, window: Window) -> subprocess.Popen:
  set_params_enabled()
  params = Params()
  params.remove("CarParamsPersistent")
  params.remove("CarParamsPrevRoute")
  params.put_bool("AlphaLongitudinalEnabled", window.alpha_long)
  env = os.environ | {'TESTING_CLOSET': '1', 'BLOCK': 'soundd'}
  # card fingerprints from the CAN and the route's firmware. without firmware it would query the UP, which doesn't answer, so the car is given
  env.pop('FINGERPRINT')
  if window.cache is None:
    env |= {'FINGERPRINT': str(route.car_model), 'SKIP_FW_QUERY': '1'}
  return subprocess.Popen([sys.executable, os.path.join(BASEDIR, "openpilot/system/manager/manager.py")], env=env, start_new_session=True)


def stop_manager(manager: subprocess.Popen) -> None:
  for sig, timeout in ((signal.SIGINT, 60), (signal.SIGKILL, None)):
    try:
      os.killpg(manager.pid, sig)
      manager.wait(timeout)
      return
    except ProcessLookupError:
      return
    except subprocess.TimeoutExpired:
      pass


def check_manager(manager: subprocess.Popen) -> None:
  if manager.poll() is not None:
    raise RuntimeError(f"manager died: rc {manager.returncode}")


def drive(up, route: CarTestRoute, window: Window) -> list:
  sm = messaging.SubMaster(['pandaStates'])
  socks = [messaging.sub_sock(s, conflate=False) for s in RECORD]
  manager = start_manager(route, window)
  try:
    up_call(up, 'reset')
    up_call(up, 'set_ignition', True)
    end = time.monotonic() + PANDA_LIMIT
    while not len(sm['pandaStates']):
      check_manager(manager)
      if time.monotonic() > end:
        raise RigProblem("rig problem: panda not answering")
      sm.update(10)
    # manager start clears CarParamsCache, card reads it on the first CAN
    if window.cache is not None:
      Params().put("CarParamsCache", window.cache)
    # the station's lanes start classic, a CAN FD bus gets the data speed the DUT's panda is set to
    panda = sm['pandaStates'][0]
    can_states = (panda.canState0, panda.canState1, panda.canState2)
    for bus in sorted({bus for frame in window.frames for _, dat, bus in frame if len(dat) > 8}):
      up_call(up, 'set_can_data_speed_kbps', bus, can_states[bus].canDataSpeed)

    for sock in socks:
      messaging.drain_sock_raw(sock)
    # once, a second call plays a second loop, the window ends before it wraps
    up_call(up, 'can_replay', window.frames)
    msgs = []
    end = time.monotonic() + len(window.frames) * DT
    while time.monotonic() < end:
      check_manager(manager)
      for sock in socks:
        msgs += messaging.drain_sock(sock)
      time.sleep(DT)
  finally:
    try:
      up_call(up, 'reset')
    finally:
      stop_manager(manager)

  seen = {m.which() for m in msgs}
  if 'cabinCameraState' not in seen or not seen & set(ROAD_CAMERAS):
    raise RigProblem("rig problem: cameras not streaming")
  return msgs


def end_manager_output() -> None:
  # the manager's last line ends without a newline and leaves its color on
  sys.stderr.flush()
  sys.stdout.write("\x1b[0m\n")
  sys.stdout.flush()


def run(up, route: CarTestRoute) -> tuple[str, Window | None, list[str]]:
  window = None
  try:
    window = find_window(route)
    checks = check(drive(up, route, window), str(route.car_model), window.edge)
    status = "passed" if all(ok for ok, _ in checks) else "failed"
    lines = [f"{'✓' if ok else '✗'} {text}" for ok, text in checks]
  except RigProblem as e:
    status, lines = "rig problem", [str(e)]
  except Exception as e:
    status, lines = "failed", [repr(e)]
  if window is not None:
    edge = f", engage edge at {window.start + LEAD:.1f} s" if window.edge else ", engagement not tested"
    edge += ", fingerprint given, the route has no firmware card can use" if window.cache is None else ""
    lines.insert(0, f"route {route.route} segment {window.segment}, window {window.start:.1f}-{window.start + window.seconds:.1f} s{edge}")
  return status, window, lines


def main() -> None:
  from ultimate_provisioning import UP  # only on a DUT
  shard, shards = int(os.getenv("ALFRED_SHARD", "0")), int(os.getenv("ALFRED_SHARDS", "1"))
  job = f"cars{shard}" if "ALFRED_SHARDS" in os.environ else "cars"
  selected = [s.strip() for s in os.getenv("SCENARIOS", "").split(",") if s.strip()]
  todo = sorted((r for r in routes if not selected or str(r.car_model) in selected or r.route in selected), key=lambda r: r.route)
  todo = [r for i, r in enumerate(todo) if i % shards == shard]

  up = UP.remote()
  start = time.monotonic()
  counts = {"passed": 0, "failed": 0, "rig problem": 0}
  for route in todo:
    print(f"=== {route.car_model} {route.route} ===", flush=True)
    t = time.monotonic()
    status, window, lines = run(up, route)
    counts[status] += 1
    end_manager_output()
    untested = "  engagement not tested" if window is not None and not window.edge else ""
    print(f"{'✅' if status == 'passed' else '❌'} {route.car_model} {route.route}{untested}  {time.monotonic() - t:.0f} s")
    if status != "passed":
      print("\n".join(f"    {line}" for line in lines))
      print(f"    repeat: SCENARIOS={route.route} ./openpilot/selfdrive/test/test_cars.py", flush=True)
    if status == "rig problem":
      break

  end_manager_output()
  rig = f", {counts['rig problem']} rig problem" if counts['rig problem'] else ""
  print(f"{job}: {counts['passed']} passed, {counts['failed']} failed{rig}, {(time.monotonic() - start) / 60:.1f} min", flush=True)
  sys.exit(counts["passed"] != len(todo))


if __name__ == "__main__":
  main()
