#!/usr/bin/env python3
"""Decodes messages with data/message.js and compares them with pycapnp's to_dict().

usage: message.py logs... [--step n]
"""
import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from openpilot.tools.lib.logreader import LogReader

HERE = Path(__file__).parent


def norm(x):
  if isinstance(x, float):
    return "nan" if math.isnan(x) else "inf" if x == math.inf else "-inf" if x == -math.inf else x
  if isinstance(x, bytes):
    return list(x)
  if isinstance(x, dict):
    return {k: norm(v) for k, v in x.items()}
  if isinstance(x, list):
    return [norm(v) for v in x]
  return x


def diffs(py, js, path=""):
  # a null pointer is null in JS; pycapnp reads it as the default struct or an empty list
  if js is None and (py == [] or isinstance(py, dict)) or py is None and js == "":
    return
  # a generic (Map entry) byte list decodes as text when NUL-terminated
  if isinstance(py, list) and isinstance(js, str) and bytes(py).rstrip(b"\0") == js.encode():
    return
  if isinstance(py, dict) and isinstance(js, dict):
    for k in py.keys() | js.keys():
      yield from diffs(py.get(k), js.get(k), f"{path}/{k}")
  elif isinstance(py, list) and isinstance(js, list) and len(py) == len(js):
    for i, (a, b) in enumerate(zip(py, js, strict=True)):
      yield from diffs(a, b, f"{path}/{i}")
  # floats match to f32 precision, and 64-bit integers to a double's
  elif py != js and not (type(py) in (int, float) and type(js) in (int, float) and math.isclose(py, js, rel_tol=1e-6)):
    yield path, py, js


def run(log, step):
  with tempfile.NamedTemporaryFile(suffix=".json") as f:
    subprocess.run(["node", HERE / "message.mjs", log, f.name, str(step)], check=True)
    js = dict(json.load(f))
  bad, n = {}, 0
  for i, ev in enumerate(LogReader(log)):
    if i not in js:
      continue
    n += 1
    try:
      py = norm(ev.to_dict())
    except Exception:  # an unknown union member
      continue
    for path, a, b in diffs(py, js[i]):
      bad.setdefault(path, (i, a, b))
  print(f"{log}: {n} messages, {len(bad)} mismatching paths")
  for path, (i, a, b) in list(bad.items())[:40]:
    print(f"  #{i} {path}: py {str(a)[:80]} js {str(b)[:80]}")
  return not bad


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("logs", nargs="+")
  p.add_argument("--step", type=int, default=1)
  args = p.parse_args()
  results = [run(log, args.step) for log in args.logs]
  sys.exit(0 if all(results) else 1)
