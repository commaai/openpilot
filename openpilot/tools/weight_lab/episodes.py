#!/usr/bin/env python3
"""List episodes where the traffic-control circuit fires, per segment npz.

usage: episodes.py NPZ [NPZ ...]
"""
import sys
import numpy as np

CIRCUIT = [69, 248, 201, 239]
THRESH = 0.3

for p in sys.argv[1:]:
  d = np.load(p)
  c = d['x_plan_trunk'][:, CIRCUIT].sum(1)
  on = np.flatnonzero(c > THRESH)
  if len(on) == 0:
    print(d['seg'][0], 'no episodes')
    continue
  for e in np.split(on, np.flatnonzero(np.diff(on) > 20) + 1):
    i = e[np.argmax(c[e])]
    print(f"{d['seg'][0]} frames {d['frame'][e[0]]}-{d['frame'][e[-1]]} peak f{d['frame'][i]} {c[i]:.2f} | v {d['v_ego'][e[0]]:.1f}->{d['v_ego'][e[-1]]:.1f} accel@peak {d['accel'][i]:+.2f} min accel {d['accel'][e].min():+.2f}")
