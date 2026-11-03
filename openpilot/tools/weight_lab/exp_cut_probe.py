#!/usr/bin/env python3
"""Baselines and gate probes for the 2-layer corner-cut circuit, held out by route.

Prints: inside offset and steering change of master vs shipped stack, then leave-one-route-out
ridge probes from x_plan_feat for lane confidence, speed, lateral accel and the offset.

usage: exp_cut_probe.py
"""
import numpy as np

from openpilot.tools.weight_lab import fit_lat_edit as F
from openpilot.tools.weight_lab.exp_cut_data import load
from openpilot.tools.weight_lab.cut import X_IDXS

LAT_T = 0.25  # same as fit_lat_edit.steer_curv
MIN_STABLE_DELAY = 0.3
MIN_SPEED = 1.0


def steer_curv(plan, v):
  """Vectorized fit_lat_edit.steer_curv (lat_t 0.25 s, no plan distance)."""
  psi = (LAT_T / MIN_STABLE_DELAY) * interp_rows(plan[:, :, 11], MIN_STABLE_DELAY)
  vc = np.maximum(v, MIN_SPEED)
  return 2 * psi / (vc * LAT_T) - plan[:, 0, 14] / vc


def interp_rows(rows, t):
  i = np.searchsorted(F.T, t) - 1
  w = (t - F.T[i]) / (F.T[i + 1] - F.T[i])
  return rows[:, i] * (1 - w) + rows[:, i + 1] * w


def y_at(plan, x=20.0):
  """Plan y at distance x for each frame."""
  px, py = plan[:, :, 0], plan[:, :, 1]
  i = np.clip((px < x).sum(1) - 1, 0, 31)
  n = np.arange(len(px))
  w = np.clip((x - px[n, i]) / np.maximum(px[n, i + 1] - px[n, i], 1e-6), 0, 1)
  return py[n, i] * (1 - w) + py[n, i + 1] * w


def inside20(plan, D):
  c20 = np.array([np.interp(20.0, X_IDXS, c) for c in D['center']])
  return np.sign(D['curv']) * (y_at(plan) - c20)


def ridge(X, y, lam, w=None):
  w = np.ones(len(X)) if w is None else w
  m, sd = np.average(X, 0, w), X.std(0) + 1e-6
  Z = (X - m) / sd
  ym = np.average(y, weights=w)
  Zw = Z * w[:, None]
  coef = np.linalg.solve(Zw.T @ Z + lam * w.sum() * np.eye(Z.shape[1]), Zw.T @ (y - ym))
  return coef / sd, ym - m @ (coef / sd)


def q(a):
  return f'p99 {np.percentile(a, 99):.4f} p99.9 {np.percentile(a, 99.9):.4f} max {a.max():.4f}'


def main():
  D = load()
  v, la, prob = D['v'], D['curv'] * D['v'] ** 2, D['lane_prob']
  routes = sorted(set(D['route']))
  print(f'{len(v)} frames, {len(set(D["seg"]))} segments, {len(routes)} routes')
  curves = (prob > 0.5) & (v > 8) & (np.abs(la) > 0.7)
  for name, pl in (('master', D['plan']), ('shipped', D['plan_ship'])):
    ins = inside20(pl, D)[curves]
    print(f'{name}: confident curves n={curves.sum()} inside@20m median {np.median(ins):+.3f} p90 {np.percentile(ins, 90):+.3f}')
  k0, k1 = steer_curv(D['plan'], v), steer_curv(D['plan_ship'], v)
  print('check vs fit_lat_edit.steer_curv:', np.abs(k0[:2000] - F.steer_curv(D['plan'][:2000], v[:2000])).max())
  dk = np.abs(k1 - k0)
  lo = (prob < 0.5) & (v > 3)
  print(f'shipped vs master |dk|: not confident {q(dk[lo])} | v<8 {q(dk[v < 8])} | straight {q(dk[(np.abs(la) < 0.3) & (v > 8)])}')
  # probes, held out by route
  X = D['x'].astype(np.float64)
  logit = np.clip(np.log(prob / (1 - prob + 1e-9) + 1e-9), -8, 8)
  off = y_at(D['plan_ship']) - np.array([np.interp(20.0, X_IDXS, c) for c in D['center']])
  targets = {'lane_logit': (logit, np.ones(len(v), bool)), 'v': (v, np.ones(len(v), bool)),
             'lat_acc': (la, v > 3), 'curv': (D['curv'] * 100, v > 3),
             'off20_ship': (off, (prob > 0.5) & (v > 8) & (np.abs(off) < 2.5))}
  for tname, (y, use) in targets.items():
    pred = np.zeros(len(v))
    for r in routes:
      tr = (D['route'] != r) & use
      w, b = ridge(X[tr], y[tr], 1e-2)
      te = D['route'] == r
      pred[te] = X[te] @ w + b
    e = pred[use] - y[use]
    print(f'{tname}: held-out R2 {1 - np.mean(e ** 2) / np.var(y[use]):.3f} | abs err p50 {np.median(np.abs(e)):.3f} p99 {np.percentile(np.abs(e), 99):.3f} max {np.abs(e).max():.3f}')
    np.save(f'/home/batman/xx/junk/shane/weight_edit/exp/cut/probe_{tname}.npy', pred)


if __name__ == '__main__':
  main()
