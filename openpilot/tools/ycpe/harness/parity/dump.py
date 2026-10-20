#!/usr/bin/env python3
"""Dumps every leaf of a log through pycapnp, named like data/capnp.js columns:
{service: {t, valid, cols: {"/field/sub": [value or None per event]}}}. Enums are ordinals, lists of at most
16 items split per index, NaN is None and infinities are "inf"/"-inf"."""
import json
import math
import sys

import capnp

from openpilot.cereal import log
from openpilot.tools.lib.logreader import LogReader

MAX_LIST = 16
SCALARS = {"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64", "enum"}
plans: dict[int, list] = {}


def scalar(v):
  if isinstance(v, capnp.lib.capnp._DynamicEnum):
    return v.raw
  if isinstance(v, float):
    return None if math.isnan(v) else ("inf" if v > 0 else "-inf") if math.isinf(v) else v
  return int(v)


def plan(schema):
  # per struct: (name, union member?, kind, element kind for lists)
  key = schema.node.id
  if key not in plans:
    p = []
    for name, f in schema.fields.items():
      member = f.proto.discriminantValue != 0xffff
      if f.proto.which() == "group":
        p.append((name, member, "group", None))
        continue
      t = f.proto.slot.type
      kind = t.which()
      kind = "scalar" if kind in SCALARS else kind
      elem = t.list.elementType.which() if kind == "list" else None
      if kind in ("scalar", "text", "struct", "list"):
        p.append((name, member, kind, "scalar" if elem in SCALARS else elem))
    plans[key] = p
  return plans[key]


def walk(s, path, cols, i):
  which = s.which() if s.schema.union_fields else None
  for name, member, kind, elem in plan(s.schema):
    if member and name != which:
      continue
    sub = f"{path}/{name}"
    if kind == "scalar":
      put(cols, sub, i, scalar(getattr(s, name)))
    elif kind == "group":
      walk(getattr(s, name), sub, cols, i)
    elif kind == "text":
      put(cols, sub, i, getattr(s, name) if s._has(name) else "")
    elif kind == "struct":
      if s._has(name):
        walk(getattr(s, name), sub, cols, i)
    elif kind == "list" and s._has(name):
      items(getattr(s, name), elem, sub, cols, i)


def items(lst, elem, path, cols, i):
  if len(lst) > MAX_LIST:
    return
  for j, x in enumerate(lst):
    if elem == "struct":
      walk(x, f"{path}/{j}", cols, i)
    elif elem == "scalar":
      put(cols, f"{path}/{j}", i, scalar(x))
    elif elem == "text":
      put(cols, f"{path}/{j}", i, x)


def put(cols, path, i, v):
  c = cols.get(path)
  if c is None:
    c = cols[path] = {}
  c[i] = v


def dump(events):
  services = {k[0]: k for k in plan(log.Event.schema)}
  out = {}
  for e in events:
    try:
      which = e.which()
    except capnp.KjException:
      continue  # newer than our schema
    svc = out.setdefault(which, {"t": [], "valid": [], "cols": {}})
    i = len(svc["t"])
    svc["t"].append(e.logMonoTime / 1e9)
    svc["valid"].append(int(e.valid))
    _, _, kind, elem = services.get(which, (None, None, "data", None))
    if kind == "struct":
      walk(getattr(e, which), "", svc["cols"], i)
    elif kind == "text":
      put(svc["cols"], "", i, getattr(e, which))
    elif kind == "list":
      items(getattr(e, which), elem, "", svc["cols"], i)
  for svc in out.values():
    n = len(svc["t"])
    svc["cols"] = {p: [c.get(i) for i in range(n)] for p, c in svc["cols"].items()}
  return out


if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("usage: dump.py <log> <out.json>")
    sys.exit(1)
  with open(sys.argv[2], "w") as f:
    json.dump(dump(LogReader(sys.argv[1])), f, separators=(",", ":"))
