#!/usr/bin/env python3
# replays the car test routes through BASE's and this opendbc's car code and safety, reports every changed safety mode and param,
# RX and TX result and safety state value
#   diff_safety.py [BASE]          BASE defaults to HEAD^, which is master on a PR's merge commit
#   diff_safety.py check [BASE]    prints changed=true when opendbc differs from BASE's
import inspect
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from opendbc.car import DT_CTRL as DT

# GitHub's comment limit is 65536
MAX_REPORT = 50000
# the panda's CAN buses, a log's can src is bus + 4 * panda
BUSES = 3
# the panda marks sent CAN with src + 128 and blocked with src + 192
TX = 128


def load(route, seconds: float | None = None, tx: bool = False):
  from opendbc.car.car_helpers import interfaces
  from opendbc.car.tests.test_models import TestCarModelBase

  R = type("R", (TestCarModelBase,), {"test_route": route, "platform": route.car_model})
  # older opendbc reads the route from test_route
  car_fw, can_msgs, alpha_long = R.get_testing_data(route) if inspect.signature(R.get_testing_data).parameters else R.get_testing_data()
  CP = interfaces[R.platform].get_params(R.platform, R.fingerprint, car_fw, alpha_long, False, docs=False)
  cfg = CP.safetyConfigs[-1]

  t0 = can_msgs[0][0]
  steps = int((can_msgs[-1][0] - t0) / 1e9 / DT) + 1 if seconds is None else int(seconds / DT)
  frames = [[] for _ in range(steps)]
  for t, msgs in can_msgs:
    i = int((t - t0) / 1e9 / DT)
    if i >= len(frames):
      break
    for m in msgs:
      if m.src < TX and m.src % 4 < BUSES:
        frames[i].append((m.address, bytes(m.dat), m.src % 4))
      # sent and blocked TX, so a check that allows more shows up too
      elif tx and m.src >= TX and m.src % 4 < BUSES:
        frames[i].append((m.address, bytes(m.dat), TX + m.src % 4))
  return cfg.safetyModel.raw, cfg.safetyParam, frames


def replay(mode: int, param: int, frames, on_tick=None):
  # the result of each message and the whole safety state (safety_a) after each 10 ms step
  from opendbc.safety.tests.libsafety import libsafety_py
  safety = libsafety_py.libsafety
  assert safety.set_safety_hooks(mode, param) == 0
  safety.set_alternative_experience(0)
  # older opendbc names it safety_tick_current_safety_config
  tick = safety.safety_tick if hasattr(safety, "safety_tick") else safety.safety_tick_current_safety_config
  state = libsafety_py.ffi.buffer(libsafety_py.ffi.addressof(safety, "safety_a"))

  steps = []
  for i, frame in enumerate(frames):
    safety.set_timer(int(i * DT * 1e6))
    if i % int(1 / DT) == 0:
      tick()
      if on_tick is not None:
        on_tick(i, safety)
    results = []
    for addr, dat, bus in frame:
      if bus >= TX:
        results.append(bool(safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus - TX, dat))))
      else:
        fwd = safety.safety_fwd_hook(bus, addr)
        results.append((bool(safety.safety_rx_hook(libsafety_py.make_CANPacket(addr, bus, dat))), fwd))
    steps.append((results, state[:]))
  return steps


def state_layout(ffi):
  # every field of the safety state with its offset and type, pointers left out since they differ between processes
  fields = []

  def walk(t, path, offset):
    if t.kind == "struct":
      for name, f in t.fields:
        walk(f.type, f"{path}.{name}" if path else name, offset + f.offset)
    elif t.kind == "array":
      for i in range(t.length):
        walk(t.item, f"{path}[{i}]", offset + i * ffi.sizeof(t.item))
    elif t.kind in ("primitive", "enum"):
      size = ffi.sizeof(t)
      if t.cname == "_Bool":
        dtype = "?"
      else:
        # cffi only converts float types with float()
        try:
          float(ffi.cast(t, 0))
          dtype = f"f{size}"
        except TypeError:
          dtype = f"i{size}" if int(ffi.cast(t, -1)) < 0 else f"u{size}"
      fields.append((path.removeprefix("state."), offset, size, dtype))

  walk(ffi.typeof("SafetyCopy"), "", 0)
  return fields


def cdata_fields(ffi, c, path: str) -> dict:
  # every value of a struct or array by name, pointers left out
  if not isinstance(c, ffi.CData):
    return {path: c}
  t = ffi.typeof(c)
  if t.kind == "struct":
    return {k: v for name, _ in t.fields for k, v in cdata_fields(ffi, getattr(c, name), f"{path}.{name}").items()}
  if t.kind == "array":
    return {k: v for i in range(t.length) for k, v in cdata_fields(ffi, c[i], f"{path}[{i}]").items()}
  return {}


def safety_config(ffi, safety) -> tuple[list[str], dict]:
  # the RX checks and allowed TX messages, named by message so a reordered config isn't a change
  cfg = safety.safety_a.setup.current_safety_config
  rx = [f"0x{cfg.rx_checks[i].msg[0].addr:x} bus {cfg.rx_checks[i].msg[0].bus}" for i in range(cfg.rx_checks_len)]
  config = {"disable_forwarding": cfg.disable_forwarding}
  for i, name in enumerate(rx):
    config |= cdata_fields(ffi, cfg.rx_checks[i], f"rx_checks[{name}]")
  for i in range(cfg.tx_msgs_len):
    config |= cdata_fields(ffi, cfg.tx_msgs[i], f"tx_msgs[0x{cfg.tx_msgs[i].addr:x} bus {cfg.tx_msgs[i].bus}]")
  return rx, config


def results_route(args):
  # with this side's car code and safety: the safety mode and param, every message's result (RX valid or TX allowed),
  # where each RX is forwarded, and each safety state field after each step
  route, out, build = args
  name = f"{route.car_model} - {route.route}"
  path = out / f"{re.sub(r'[^\w.-]', '_', f'{name} {route.segment}')}.npz"
  try:
    mode, param, frames = load(route, tx=True)
  except Exception as e:
    np.savez(path, name=name, error=f"can't load, {e!r}")
    return
  from opendbc.safety.tests.libsafety import libsafety_py
  # a copy of the build is a fresh library, no safety state carries over from another route
  fd, copy = tempfile.mkstemp(suffix=".so")
  os.close(fd)
  shutil.copy(build, copy)
  libsafety_py.load(copy)
  os.unlink(copy)
  arrays: dict[str, Any] = {"name": np.array(name), "config:safety_mode": np.array(mode), "config:safety_param": np.array(param)}
  if libsafety_py.libsafety.set_safety_hooks(mode, param) != 0:
    np.savez(path, **arrays)
    return
  steps = replay(mode, param, frames)
  results = [r for step_results, _ in steps for r in step_results]
  raw = np.frombuffer(b"".join(state for _, state in steps), dtype=np.uint8).reshape(len(steps), -1)
  rx, config = safety_config(libsafety_py.ffi, libsafety_py.libsafety)
  arrays |= {f"config:{field}": np.array(v) for field, v in config.items()}
  for field, offset, size, dtype in state_layout(libsafety_py.ffi):
    # an RX check's status by its message, unused slots left out
    if field.startswith("rx_status["):
      i, rest = int(field[len("rx_status["):field.index("]")]), field[field.index("]") + 1:]
      if i >= len(rx):
        continue
      field = f"rx_status[{rx[i]}]{rest}"
    arrays[f"state:{field}"] = raw[:, offset:offset + size].copy().view(dtype)[:, 0]
  events = np.array([(i, addr, bus) for i, frame in enumerate(frames) for addr, _, bus in frame], dtype=np.int64).reshape(-1, 3)
  arrays["frame"], arrays["addr"], arrays["bus"] = events.T
  arrays["ok"] = np.array([r[0] if isinstance(r, tuple) else r for r in results], dtype=bool)
  arrays["fwd"] = np.array([r[1] if isinstance(r, tuple) else -2 for r in results], dtype=np.int16)
  np.savez_compressed(path, **arrays)


def pool():
  return ProcessPoolExecutor(os.cpu_count(), mp_context=multiprocessing.get_context("spawn"))


def value(v):
  return bool(v) if isinstance(v, np.bool_) else v.item()


def message_diffs(a, b) -> list[str]:
  # RX valid, RX forward and TX allowed, grouped by message and change
  changes = Counter()
  first = {}
  for k in np.flatnonzero((a["ok"] != b["ok"]) | (a["fwd"] != b["fwd"])):
    i, addr, bus = int(a["frame"][k]), int(a["addr"][k]), int(a["bus"][k])
    if bus >= TX:
      key = (f"TX 0x{addr:x} bus {bus - TX} allowed", bool(a["ok"][k]), bool(b["ok"][k]))
    elif a["ok"][k] != b["ok"][k]:
      key = (f"RX 0x{addr:x} bus {bus} valid", bool(a["ok"][k]), bool(b["ok"][k]))
    else:
      key = (f"RX 0x{addr:x} bus {bus} forward", int(a["fwd"][k]), int(b["fwd"][k]))
    changes[key] += 1
    first.setdefault(key, i)
  return [f"\n  {what}: {old} -> {new} ({n} msgs, from frame {first[(what, old, new)]})" for (what, old, new), n in sorted(changes.items())]


class StateWrap:
  """Adapter so car_diff reads one safety state field, dotted names are nested like in a struct"""
  def __init__(self, name: str, v):
    *parents, leaf = name.split(".")
    self._d = d = {}
    for p in parents:
      d = d.setdefault(p, {})
    d[leaf] = v
  def to_dict(self) -> dict:
    return self._d


def config_diffs(a, b) -> list[str]:
  lines = []
  for key in sorted(k for k in a.keys() | b.keys() if k.startswith("config:")):
    name = key.removeprefix("config:")
    if key not in a or key not in b:
      lines.append(f"\n  config {name}: only in {'master' if key in a else 'this PR'}")
    elif a[key] != b[key]:
      lines.append(f"\n  config {name}: {value(a[key])} -> {value(b[key])}")
  return lines


def state_diffs(a, b, full: bool = False) -> list[str]:
  # full: every changed frame instead of the first ones, for the report in the artifact
  from opendbc.car.tests.car_diff import format_diff, format_numeric_diffs
  lines = []
  for key in sorted(a.keys() | b.keys()):
    if not key.startswith("state:"):
      continue
    name = key.removeprefix("state:")
    if key not in a or key not in b:
      lines.append(f"\n  {name}: only in {'master' if key in a else 'this PR'}")
      continue
    old, new = a[key], b[key]
    frames = np.flatnonzero(old != new)
    if not len(frames):
      continue
    diffs = [(name, int(i), (value(old[i]), value(new[i])), int(i * DT * 1e9)) for i in frames]
    ref = [(int(i * DT * 1e9), StateWrap(name, value(v))) for i, v in enumerate(old)]
    states = [StateWrap(name, value(v)) for v in new]
    lines.append(f"\n  {name} ({len(diffs)} diffs)")
    if full and not isinstance(diffs[0][2][0], bool):
      for i in range(0, len(diffs), 10):
        lines += format_numeric_diffs(diffs[i:i + 10])
    else:
      lines += format_diff(diffs, ref, states, name)
  return lines


def diff_report(base: Path, new: Path) -> int:
  changed, changed_full, errors, n_passed = [], [], [], 0
  for key in sorted({p.name for p in base.glob("*.npz")} | {p.name for p in new.glob("*.npz")}):
    a = dict(np.load(base / key)) if (base / key).exists() else None
    b = dict(np.load(new / key)) if (new / key).exists() else None
    name = str(np.load(new / key if b is not None else base / key)["name"])
    if a is None or b is None:
      lines = ["", f"  {'new test route' if a is None else 'test route removed'}"]
      changed.append((name, lines))
      changed_full.append((name, lines))
    elif "error" in a or "error" in b:
      errors += [f"{name}: {side}, {d['error']}" for side, d in (("master", a), ("this PR", b)) if "error" in d]
    elif "ok" not in b:
      errors.append(f"{name}: this opendbc has no safety mode {b['config:safety_mode']}")
    elif "ok" not in a:
      lines = ["", f"  new safety mode {b['config:safety_mode']}, master's opendbc doesn't have it"]
      changed.append((name, lines))
      changed_full.append((name, lines))
    elif not all(np.array_equal(a[k], b[k]) for k in ("frame", "addr", "bus")):
      errors.append(f"{name}: the replayed CAN differs between master and this PR")
    else:
      messages, config, states = message_diffs(a, b), config_diffs(a, b), state_diffs(a, b)
      if messages or config or states:
        changed.append((name, messages + config + states))
        changed_full.append((name, messages + config + state_diffs(a, b, full=True)))
      else:
        n_passed += 1

  from openpilot.common.basedir import BASEDIR
  from opendbc.car.tests.car_diff import format_report

  description = "Replays every car test route through this PR's panda safety and compares the behavior to master."
  full = format_report("Safety replay diff report", description, changed_full, n_passed, errors, max_len=None)
  (Path(BASEDIR) / "diff_safety_full.txt").write_text(full)
  # the comment shows as many routes as fit, the full report is in the job's artifact
  report = format_report("Safety replay diff report", description, changed, n_passed, errors, max_len=MAX_REPORT)
  print(report)
  (Path(BASEDIR) / "diff_safety.txt").write_text(report)
  return int(bool(changed or errors))


def main() -> int:
  # one side, with its opendbc on PYTHONPATH
  if sys.argv[1:2] == ["results"]:
    out = Path(sys.argv[2])
    out.mkdir()
    from opendbc.car.tests.routes import routes
    from opendbc.safety.tests.libsafety import libsafety_py
    build = libsafety_py._build_libsafety()
    try:
      with pool() as p:
        list(p.map(results_route, [(r, out, build) for r in routes]))
    finally:
      os.unlink(build)
    return 0

  from openpilot.common.basedir import BASEDIR
  root = Path(BASEDIR)

  def git(*args, cwd=root):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()

  check = sys.argv[1:2] == ["check"]
  base_ref = (sys.argv[2:] if check else sys.argv[1:]) or ["HEAD^"]
  base, new = git("rev-parse", f"{base_ref[0]}:opendbc_repo"), git("rev-parse", "HEAD:opendbc_repo")
  # for a GitHub step's outputs
  if check:
    print(f"changed={str(base != new).lower()}")
    return 0
  if base == new:
    print(f"opendbc is unchanged ({new}), nothing to compare")
    return 0
  print(f"comparing opendbc {base} -> {new}", flush=True)

  with tempfile.TemporaryDirectory() as tmp:
    w, odbc = Path(tmp), root / "opendbc_repo"
    git("fetch", "-q", "--depth=1", "origin", base, cwd=odbc)
    git("worktree", "add", "-q", "--detach", str(w / "base"), base, cwd=odbc)
    try:
      for name, path in (("base", w / "base"), ("new", odbc)):
        subprocess.check_call([sys.executable, __file__, "results", w / f"{name}_results"], env={**os.environ, "PYTHONPATH": str(path)})
      return diff_report(w / "base_results", w / "new_results")
    finally:
      git("worktree", "remove", "--force", str(w / "base"), cwd=odbc)


if __name__ == "__main__":
  sys.exit(main())
