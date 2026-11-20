#!/usr/bin/env python3
# one test route per safety mode and param on a DUT in an ultimate provisioning (UP) station, the UP plays its engagement's CAN raw to the segment's end.
# ALFRED_SHARD/ALFRED_SHARDS drive every Nth safety mode, SCENARIOS=TOYOTA_COROLLA_TSS2,HONDA_CIVIC only those car models or routes
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import openpilot.cereal.messaging as messaging
from opendbc.car.car_helpers import interfaces
from opendbc.car.tests.routes import CarTestRoute, routes
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.selfdrive.test.safety_modes.route import DT, LEAD, Window, find_window
from openpilot.selfdrive.test.safety_modes.rules import check
from openpilot.selfdrive.test.helpers import set_params_enabled

PANDA_LIMIT = 45.
# seconds recorded after the car's CAN stops, the panda drops controls within 2 s
QUIET = 3.
ROAD_CAMERAS = ('narrowRoadCameraState', 'wideRoadCameraState')
RECORD = ('can', 'pandaStates', 'selfdriveState', 'carParams', 'cabinCameraState', *ROAD_CAMERAS)


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
  # card fingerprints from the CAN and the route's firmware, and gets the car when the firmware doesn't identify it.
  # without firmware it would query the UP, which doesn't answer
  env.pop('FINGERPRINT')
  if window.given:
    env['FINGERPRINT'] = str(route.car_model)
  if window.cache is None:
    env['SKIP_FW_QUERY'] = '1'
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


def drive(up, route: CarTestRoute, window: Window, times: dict[str, float]) -> list:
  sm = messaging.SubMaster(['pandaStates'])
  socks = [messaging.sub_sock(s, conflate=False) for s in RECORD]
  t = time.monotonic()
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
    times['panda'] = time.monotonic() - t
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
    # once, a second call plays a second loop. then QUIET s without the car's CAN, ignition still on, the drive ends before it wraps
    frames = window.frames + [[] for _ in range(round(QUIET / DT))]
    up_call(up, 'can_replay', frames)
    times['replay_start'] = t = time.monotonic()
    msgs = []
    end = time.monotonic() + len(frames) * DT
    while time.monotonic() < end:
      check_manager(manager)
      for sock in socks:
        msgs += messaging.drain_sock(sock)
      time.sleep(DT)
    times['replay'] = time.monotonic() - t
  finally:
    t = time.monotonic()
    try:
      up_call(up, 'reset')
    finally:
      stop_manager(manager)
    times['stop'] = time.monotonic() - t

  seen = {m.which() for m in msgs}
  if 'cabinCameraState' not in seen or not seen & set(ROAD_CAMERAS):
    raise RigProblem("rig problem: cameras not streaming")
  return msgs


def end_manager_output() -> None:
  # the manager's last line ends without a newline and leaves its color on
  sys.stderr.flush()
  sys.stdout.write("\x1b[0m\n")
  sys.stdout.flush()


def safety_groups(selected: list[str]) -> dict[tuple[str, int], list[CarTestRoute]]:
  # the test routes of each safety mode and param opendbc gives their platforms, dashcamOnly ones aren't driven
  groups: dict[tuple[str, int], list[CarTestRoute]] = {}
  for route in sorted(routes, key=lambda r: r.route):
    if route.car_model is None or (selected and str(route.car_model) not in selected and route.route not in selected):
      continue
    CP = interfaces[route.car_model].get_non_essential_params(route.car_model)
    if not CP.dashcamOnly:
      cfg = CP.safetyConfigs[-1]
      groups.setdefault((str(cfg.safetyModel), cfg.safetyParam), []).append(route)
  return dict(sorted(groups.items()))


def best_window(group: list[CarTestRoute]) -> tuple[CarTestRoute | None, Window | None]:
  # the route whose window disengages most, ties go to the first
  best: tuple[CarTestRoute | None, Window | None] = (None, None)
  for route in group:
    window = find_window(route)
    if window is not None and (best[1] is None or window.disengages > best[1].disengages):
      best = (route, window)
  return best


def first(msgs, service: str, since: float) -> str:
  t = next((m.logMonoTime * 1e-9 for m in msgs if m.which() == service), None)
  return "never" if t is None else f"+{t - since:.1f} s"


def timing(times: dict[str, float], msgs) -> str:
  # where a drive's time goes, the replay's first CAN is at +0
  steps = [f"{k} {times[k]:.0f} s" for k in ('panda', 'replay', 'stop') if k in times]
  if 'replay_start' in times:
    steps += [f"{s} {first(msgs, s, times['replay_start'])}" for s in ('carParams', 'selfdriveState')]
  return ", ".join(steps)


def run(up, route: CarTestRoute, window: Window) -> tuple[str, list[str]]:
  times: dict[str, float] = {}
  msgs = []
  try:
    msgs = drive(up, route, window, times)
    checks = check(msgs, str(route.car_model))
    status = "passed" if all(ok for ok, _ in checks) else "failed"
    lines = [f"{'✓' if ok else '✗'} {text}" for ok, text in checks]
  except RigProblem as e:
    status, lines = "rig problem", [str(e)]
  except Exception as e:
    status, lines = "failed", [repr(e)]
  given = ", fingerprint given, the route's firmware doesn't identify the car" if window.given else ""
  lines.insert(0, f"route {route.route} segment {window.segment}, engage edge at {window.start + LEAD:.1f} s, {window.disengages} disengagements{given}")
  lines.insert(1, timing(times, msgs))
  return status, lines


def main() -> None:
  from ultimate_provisioning import UP  # only on a DUT
  shard, shards = int(os.getenv("ALFRED_SHARD", "0")), int(os.getenv("ALFRED_SHARDS", "1"))
  job = f"safety_modes{shard}" if "ALFRED_SHARDS" in os.environ else "safety_modes"
  selected = [s.strip() for s in os.getenv("SCENARIOS", "").split(",") if s.strip()]
  groups = [g for i, g in enumerate(safety_groups(selected).items()) if i % shards == shard]

  start = time.monotonic()
  # parsing a route's logs takes seconds, all windows the cache doesn't have load in parallel before the first drive
  with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("spawn")) as pool:
    windows = list(pool.map(best_window, [group for _, group in groups]))
  print(f"windows of {len(groups)} safety modes loaded in {time.monotonic() - start:.0f} s", flush=True)

  up = UP.remote()
  counts = {"passed": 0, "failed": 0, "rig problem": 0, "not driven": 0}
  for ((mode, param), group), (route, window) in zip(groups, windows, strict=True):
    t = time.monotonic()
    if route is None or window is None:
      counts["not driven"] += 1
      print(f"⚪ {mode} {param}  none of its {len(group)} routes engages in its test segments", flush=True)
      continue
    print(f"=== {mode} {param} {route.car_model} {route.route} ===", flush=True)
    status, lines = run(up, route, window)
    counts[status] += 1
    end_manager_output()
    print(f"{'✅' if status == 'passed' else '❌'} {mode} {param} {route.car_model}  {time.monotonic() - t:.0f} s: {lines[1]}")
    if status != "passed":
      print("\n".join(f"    {line}" for line in lines))
      print(f"    repeat: SCENARIOS={route.route} ./openpilot/selfdrive/test/test_safety_modes.py", flush=True)
    if status == "rig problem":
      break

  end_manager_output()
  extra = "".join(f", {counts[k]} {k}" for k in ("rig problem", "not driven") if counts[k])
  print(f"{job}: {counts['passed']} passed, {counts['failed']} failed{extra}, {(time.monotonic() - start) / 60:.1f} min", flush=True)
  sys.exit(counts["failed"] + counts["rig problem"] > 0)


if __name__ == "__main__":
  main()
