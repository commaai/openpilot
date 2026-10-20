#!/usr/bin/env python3
"""Schema-era parity: the fields data/migrate.js moves back for logs from before a schema change python has no migration
for (its FIXES) against pycapnp reading the log with the schema from before the change.

usage: eras.py logs...
"""
import argparse
import json
import subprocess
import sys
import tempfile
from importlib.resources import files
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

# git revision of the old schema → [(service, field path)] it read where the current schema has moved them
ERAS = {
  "069506aa36^": [("selfdriveState", "/alertSound")],
}
FIXED = [f for fields in ERAS.values() for f in fields]  # diff.py --migrate leaves these here: python doesn't fix them


def read_old(rev, data, fields):
  """{"/service/path": [value per event of the service]} with the schema at rev. pycapnp can't load a second schema
  with the same file ids, so this runs in a process that never imports cereal."""
  import capnp
  with tempfile.TemporaryDirectory() as tmp:
    for name in ("log.capnp", "custom.capnp", "deprecated.capnp", "include/c++.capnp"):
      out = Path(tmp) / name
      out.parent.mkdir(exist_ok=True)
      out.write_bytes(subprocess.run(["git", "-C", REPO, "show", f"{rev}:openpilot/cereal/{name}"], capture_output=True, check=True).stdout)
    log = capnp.load(str(Path(tmp) / "log.capnp"), imports=[tmp, str(files("opendbc") / "car")])
  out = {f"/{svc}{path}": [] for svc, path in fields}
  for e in log.Event.read_multiple_bytes(Path(data).read_bytes()):
    for svc, path in fields:
      if e.which() == svc:
        v = getattr(e, svc)
        for k in path.split("/")[1:]:
          v = getattr(v, k)
        out[f"/{svc}{path}"].append(getattr(v, "raw", v))
  return out


def run(log):
  from openpilot.tools.lib.logreader import LogReader
  with tempfile.NamedTemporaryFile(suffix=".json") as f:
    subprocess.run(["node", HERE / "dump.mjs", log, f.name, "--migrate"], check=True)
    js = json.load(f)
  bad, n, moved = {}, 0, 0
  with tempfile.NamedTemporaryFile(suffix=".log") as raw:
    raw.write(b"".join(e.as_builder().to_bytes() for e in LogReader(log)))
    raw.flush()
    for rev, fields in ERAS.items():
      args = [sys.executable, __file__, "--old", rev, raw.name, json.dumps(fields)]
      old = json.loads(subprocess.run(args, capture_output=True, check=True, text=True).stdout)
      for path, a in old.items():
        if not a:
          continue  # not in this log (a migrated selfdriveState from controlsState is python's)
        svc = path.split("/")[1]
        b = js.get(svc, {}).get("cols", {}).get(path[len(svc) + 1:]) or []
        n += len(a)
        moved += sum(1 for x in a if x)
        if a != b:
          i = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
          bad[path] = (i, a[i:i + 1], b[i:i + 1])
  return n, moved, bad


if __name__ == "__main__":
  if sys.argv[1] == "--old":
    print(json.dumps(read_old(sys.argv[2], sys.argv[3], json.loads(sys.argv[4]))))
    sys.exit()
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("logs", nargs="+")
  args = parser.parse_args()
  failed = False
  for log in args.logs:
    n, moved, bad = run(log)
    print(f"{log}: {n} values ({moved} set), {len(bad)} mismatching paths")
    for path, (i, a, b) in bad.items():
      print(f"  {path}: first at {i}: old schema {a} js {b}")
    failed |= bool(bad)
  raise SystemExit(failed)
