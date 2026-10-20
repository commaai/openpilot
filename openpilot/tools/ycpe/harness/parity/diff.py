#!/usr/bin/env python3
"""Reader parity: dumps each log through pycapnp (dump.py) and data/capnp.js (dump.mjs) and reports mismatches by path.
With --migrate, compares process_replay migrate_all against data/migrate.js instead."""
import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.ycpe.harness.parity.dump import dump
from openpilot.tools.ycpe.harness.parity.eras import FIXED

HERE = Path(__file__).parent


def same(a, b, rtol):
  if a == b or a is None and b == "" or a == "" and b is None:  # an unset text reads as ""
    return True
  if isinstance(a, (int, float)) and isinstance(b, (int, float)):
    return float(a) == float(b) or math.isclose(a, b, rel_tol=rtol, abs_tol=rtol)
  return False


def compare(py, js, rtol=0.0):
  """returns ({"/service/path": (mismatches, first index, py value, js value)}, values compared)"""
  bad, n = {}, 0
  for name in sorted(py.keys() | js.keys()):
    p, j = py.get(name), js.get(name)
    if p is None or j is None or len(p["t"]) != len(j["t"]):
      bad[f"/{name}"] = ("events", 0, p and len(p["t"]), j and len(j["t"]))
      continue
    cols = {"@t": (p["t"], j["t"]), "@valid": (p["valid"], j["valid"])}
    for path in p["cols"].keys() | j["cols"].keys():
      cols[path] = (p["cols"].get(path), j["cols"].get(path))
    for path, (a, b) in sorted(cols.items()):
      # js drops columns that never got a value (NaN counts as none), so a missing path is all None
      a, b = a or [None] * len(p["t"]), b or [None] * len(p["t"])
      n += len(a)
      miss = [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if not same(x, y, rtol)]
      if miss:
        bad[f"/{name}{path}"] = (len(miss), miss[0], a[miss[0]], b[miss[0]])
  return bad, n


def by_time(svc):
  order = sorted(range(len(svc["t"])), key=svc["t"].__getitem__)
  return {k: [svc[k][i] for i in order] for k in ("t", "valid")} | {"cols": {p: [c[i] for i in order] for p, c in svc["cols"].items()}}


def run(log, migrate, all_migrations, rtol):
  events = list(LogReader(log))
  if migrate:
    from openpilot.selfdrive.test.process_replay.migration import migrate_all
    events = migrate_all(events, manager_states=all_migrations, panda_states=all_migrations, camera_states=all_migrations)
  py = dump(events)
  with tempfile.NamedTemporaryFile(suffix=".json") as f:
    flags = (["--migrate"] if migrate else []) + (["--all"] if all_migrations else [])
    subprocess.run(["node", HERE / "dump.mjs", log, f.name, *flags], check=True)
    js = json.load(f)
  if migrate:  # migrate_all sorts the log by logMonoTime; eras.py checks what python doesn't migrate
    js = {name: by_time(svc) for name, svc in js.items()}
    for svc, path in FIXED:
      for side in (py, js):
        side.get(svc, {}).get("cols", {}).pop(path, None)
  return py, compare(py, js, rtol)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("logs", nargs="+")
  parser.add_argument("--migrate", action="store_true")
  parser.add_argument("--all", action="store_true", help="include the opt-in manager, panda and camera state migrations")
  parser.add_argument("--rtol", type=float, default=0.0, help="tolerance for numbers")
  parser.add_argument("--max", type=int, default=40, help="mismatching paths listed per log")
  args = parser.parse_args()

  failed = False
  for log in args.logs:
    py, (bad, n) = run(log, args.migrate, args.all, args.rtol)
    print(f"{log}: {len(py)} services, {sum(len(s['cols']) for s in py.values())} paths, {n} values, {len(bad)} mismatching paths")
    for path, (count, i, a, b) in list(bad.items())[:args.max]:
      print(f"  {path}: {count} (first at {i}: py={a!r} js={b!r})")
    failed |= bool(bad)
  raise SystemExit(failed)
