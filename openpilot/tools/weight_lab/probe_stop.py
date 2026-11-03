#!/usr/bin/env python3
"""Is there an earlier "stop control ahead" signal upstream of the plan-trunk circuit?

Stops = circuit episodes that peak > 2 (camera-checked red lights and STOP lines on 000005fd). The stop point is
the min speed of the episode. Each frame gets its distance to the next stop point (integrated speed).
Positives: 30-120 m before a stop. Negatives: > 200 m from any stop, moving > 5 m/s (includes green passes).
A ridge probe per feature is scored leave-one-segment-out; the score is reported by distance band.

usage: probe_stop.py
"""
import numpy as np

from openpilot.tools.weight_lab.lab_paths import CODE, DATA as HERE  # noqa: F401
CIRCUIT = [69, 248, 201, 239]
SEGS = range(9)
BANDS = [(0, 30), (30, 50), (50, 80), (80, 120), (120, 200)]


def label(d):
  c = d['x_plan_trunk'][:, CIRCUIT].sum(1)
  v = d['v_ego']
  on = np.flatnonzero(c > 0.3)
  stops = []
  if len(on):
    for e in np.split(on, np.flatnonzero(np.diff(on) > 20) + 1):
      if c[e].max() > 2 and v[e[0]] > 3:
        stops.append(e[np.argmin(v[e])])
  pos = np.concatenate([[0], np.cumsum(v * 0.05)])[:-1]  # distance travelled
  dist = np.full(len(v), np.inf)
  for s in stops:
    before = np.arange(len(v)) <= s
    dist = np.where(before, np.minimum(dist, pos[s] - pos), dist)
  # frames after the last stop of a segment can be < 200 m before a stop in the next segment: drop them from negatives
  return dist, stops


def ridge_fit(X, y, lam):
  m, sd = X.mean(0), X.std(0) + 1e-6
  Z = (X - m) / sd
  w = np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
  return lambda A: ((A - m) / sd) @ w + y.mean()


def main():
  data = {k: np.load(f'{HERE}/seg/fd_{k}.npz') for k in SEGS}
  labels = {k: label(d) for k, d in data.items()}
  print('stops per segment:', {k: len(v[1]) for k, v in labels.items()})
  feats = {'circuit (4 units)': lambda d: d['x_plan_trunk'][:, CIRCUIT],
           'plan trunk (256)': lambda d: d['x_plan_trunk'],
           'temporal feature (512)': lambda d: d['x_plan_feat'],
           'vision feature (512)': lambda d: d['x_lead_feat']}
  for name, get in feats.items():
    scores = {b: [] for b in BANDS}
    neg_scores = []
    for test in SEGS:
      tr = [k for k in SEGS if k != test and len(labels[k][1])]
      X, y = [], []
      for k in tr:
        dist, _ = labels[k]
        v = data[k]['v_ego']
        sel = ((dist >= 30) & (dist <= 120)) | ((dist > 200) & (v > 5))
        X.append(get(data[k])[sel])
        y.append(((dist >= 30) & (dist <= 120))[sel].astype(np.float32))
      f = ridge_fit(np.concatenate(X), np.concatenate(y), lam=1.0)
      dist, _ = labels[test]
      v = data[test]['v_ego']
      s = f(get(data[test]))
      for lo, hi in BANDS:
        m = (dist >= lo) & (dist < hi) & (v > 3)
        scores[(lo, hi)].append(s[m])
      neg_scores.append(s[(dist > 200) & (v > 5)])
    neg = np.concatenate(neg_scores)
    thr = np.percentile(neg, 95)  # 5% false positive rate on open road / greens
    row = ' '.join(f'{lo}-{hi} m: {100 * np.mean(np.concatenate(scores[(lo, hi)]) > thr):4.0f}%' for lo, hi in BANDS)
    print(f'{name:>24} | detected at 5% FP: {row}')


if __name__ == '__main__':
  main()
