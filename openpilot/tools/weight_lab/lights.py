#!/usr/bin/env python3
"""Every controlled stop (stop circuit active) and the start after it, from replay npz files.

Approach: circuit onset distance before the stop, speed at onset, decel needed vs model decel.
Start: frame the circuit drops (model sees go), frame model accel > 0.5 m/s², frame the car moved.

usage: lights.py [GLOB]   (default: all base replays)
"""
import glob
import os
import sys
import numpy as np

from openpilot.tools.weight_lab.lab_paths import DATA

CIRCUIT = [69, 248, 201, 239]
SKIP = ('cut', 'g2', 'fdp', 'pcut')


def events(d):
  c = d['x_plan_trunk'][:, CIRCUIT].sum(1)
  v, a, fr = d['v_ego'], d['accel'], d['frame']
  out = []
  stops = np.flatnonzero((v[1:] < 0.5) & (v[:-1] >= 0.5)) + 1
  for i1 in stops:
    pre = slice(max(0, i1 - 200), i1)
    if c[pre].max() < 1.5 or v[pre].max() < 4:
      continue  # not a controlled stop from speed
    i0 = i1
    while i0 > 0 and (c[i0 - 1] > 0.3 or i1 - i0 < 5):
      i0 -= 1
    dist = float(np.sum(v[i0:i1]) * 0.05)
    e = dict(stop=int(fr[i1]), onset=int(fr[i0]), v_on=float(v[i0]), dist=dist, need=float(v[i0] ** 2 / 2 / max(dist, 1)),
             min_a=float(a[i0:i1].min()), peak=float(c[pre].max()))
    # start: after the car is stopped
    j = i1
    while j < len(v) and v[j] < 0.5:
      j += 1
    if j >= len(v):
      out.append(e)
      continue
    stopped = slice(i1, j)
    off = next((k for k in range(i1, j + 1) if c[k] < 0.1 and c[k - 1] >= 0.1), None)
    go = next((k for k in range(i1, min(len(a), j + 60)) if a[k] > 0.5), None)
    e.update(move=int(fr[j]), stopped_s=(j - i1) / 20, hold=float(np.median(c[stopped])),
             off=None if off is None else int(fr[off]), go=None if go is None else int(fr[go]))
    out.append(e)
  return out


def main():
  pat = sys.argv[1] if len(sys.argv) > 1 else f'{DATA}/seg/*.npz'
  n = 0
  for p in sorted(glob.glob(pat)):
    if os.path.basename(p).startswith(SKIP):
      continue
    d = np.load(p)
    for e in events(d):
      n += 1
      s = (f"{os.path.basename(p)[:-4]:>9} stop f{e['stop']:4d}: onset {e['dist']:5.1f} m @ {e['v_on']:4.1f} m/s, need {e['need']:.2f}, model {e['min_a']:+.2f} | ")
      if 'move' in e:
        lag = lambda k: '-' if k is None else f"{(k - e['move']) / 20:+.1f}s"
        s += f"stopped {e['stopped_s']:4.1f}s hold {e['hold']:.2f} | circuit off {lag(e['off'])}, model go {lag(e['go'])} (vs car moved)"
      print(s)
  print(n, 'controlled stops')


if __name__ == '__main__':
  main()
