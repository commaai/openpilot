#!/usr/bin/env python3
"""Distance from stop-circuit onset to the stop, for every stop from speed. usage: onsets.py NPZ [NPZ ...]"""
import sys
import numpy as np

CIRCUIT = [69, 248, 201, 239]
for p in sys.argv[1:]:
  d = np.load(p)
  c = d['x_plan_trunk'][:, CIRCUIT].sum(1)
  v, a, fr = d['v_ego'], d['accel'], d['frame']
  stops = np.flatnonzero((v[1:] < 0.5) & (v[:-1] >= 0.5)) + 1
  for i1 in stops:
    # walk back from the stop to the start of this braking episode's circuit firing
    i0 = i1
    while i0 > 0 and (c[i0 - 1] > 0.3 or (i1 - i0) < 5):
      i0 -= 1
    if i1 - i0 < 5 or v[i0] < 4:
      start = max(0, i1 - 200)
      print(f"{p.rsplit('/', 1)[1]} stop f{fr[i1]}: circuit not on before stop (peak in prior 10 s {c[start:i1].max():.2f}, v 10 s before {v[start]:.1f})")
      continue
    dist = np.sum(v[i0:i1]) * 0.05
    print(f"{p.rsplit('/', 1)[1]} stop f{fr[i1]}: onset f{fr[i0]} v {v[i0]:4.1f} m/s, {dist:5.1f} m, needs {v[i0] ** 2 / 2 / max(dist, 1):.2f} m/s², model min {a[i0:i1].min():+.2f}")
