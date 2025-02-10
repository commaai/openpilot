#!/usr/bin/env python3
"""
Benchmark for https://github.com/commaai/openpilot/pull/34562

On a MacBook Pro M1...

% `uv run --python=3.11 parse_struct_benchmark.py`
old_parse_structs(): 7.900648624985479 seconds
new_parse_structs(): 5.590290749911219 seconds

% `python3.13 parse_struct_benchmark.py`
old_parse_structs(): 8.46019749995321 seconds
new_parse_structs(): 6.378564580809325 seconds
"""

from timeit import timeit

from openpilot.system.qcomgpsd.structs import TYPES, parse_struct as new_parse_struct  # noqa: F401

def old_parse_struct(ss):
  st = "<"
  nams = []
  for l in ss.strip().split("\n"):
    if len(l.strip()) == 0:
      continue
    typ, nam = l.split(";")[0].split()
    #print(typ, nam)
    if typ == "float" or '_Flt' in nam:
      st += "f"
    elif typ == "double" or '_Dbl' in nam:
      st += "d"
    elif typ in ["uint8", "uint8_t"]:
      st += "B"
    elif typ in ["int8", "int8_t"]:
      st += "b"
    elif typ in ["uint32", "uint32_t"]:
      st += "I"
    elif typ in ["int32", "int32_t"]:
      st += "i"
    elif typ in ["uint16", "uint16_t"]:
      st += "H"
    elif typ in ["int16", "int16_t"]:
      st += "h"
    elif typ in ["uint64", "uint64_t"]:
      st += "Q"
    else:
      raise RuntimeError(f"unknown type {typ}")
    if '[' in nam:
      cnt = int(nam.split("[")[1].split("]")[0])
      st += st[-1]*(cnt-1)
      for i in range(cnt):
        nams.append(f'{nam.split("[")[0]}[{i}]')
    else:
      nams.append(nam)
  return st, nams


if __name__ == "__main__":
    lines = "\n".join(f"{typ} stuff" for typ in TYPES)
    print(lines)
    setup = "from __main__ import lines, old_parse_struct, new_parse_struct"
    print(f"old_parse_structs(): {timeit('old_parse_struct(lines)', setup=setup)} seconds")
    print(f"new_parse_structs(): {timeit('new_parse_struct(lines)', setup=setup)} seconds")
