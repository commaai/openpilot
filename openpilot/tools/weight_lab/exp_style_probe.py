#!/usr/bin/env python3
"""Look for a "driving style" direction in x_plan_feat (the plan head input).

Situations (all from the 111 base replay segments, 20 Hz):
  launch:  0.3 < v < 3 m/s and the car speeds up past 3 m/s in the next 3 s
  follow:  lead (radar, else model lead with prob > 0.5) at < 60 m, v > 5, |v_lead - v| < 2
  cruise:  no lead within 80 m, v > 8 m/s
Targets per situation (model side = what the plan says, real side = what the car did):
  launch:  plan dv over 2 s, real dv over 2 s
  follow:  plan time gap at 4 s, real time gap at 4 s
  cruise:  plan vx(6 s) - v, real v(t + 6 s) - v
Controls: v, v^2, desired accel, lead gap, lead v, lead present. Each target and x get the controls regressed out
(Frisch-Waugh), then a ridge probe x_res -> y_res is scored with leave-one-route-out.
A style direction must transfer: the launch probe must also predict the follow and cruise residuals.
"""
import os
import numpy as np

from openpilot.tools.weight_lab import head
from openpilot.tools.weight_lab.lab_paths import DATA

OUT = f'{DATA}/exp/style'
T = np.array([10 * (i / 32) ** 2 for i in range(33)])
HZ = 20
RIDGE = 30.0
COL_X, COL_VX, COL_AX = 0, 3, 6


def at(plan_col, t):
  """Interpolate plan columns (n, 33) at time t."""
  return np.array([np.interp(t, T, row) for row in plan_col])


def shift(a, k):
  """a[i + k] inside one segment, nan past the end."""
  out = np.full_like(a, np.nan, dtype=np.float64)
  if k < len(a):
    out[:len(a) - k] = a[k:]
  return out


def load_all():
  rows = {k: [] for k in ('x', 'plan', 'v', 'accel', 'lead_x', 'lead_v', 'route', 'real_dv2', 'real_dv6', 'v3max',
                          'real_gap4')}
  for path in head.base_files():
    d = np.load(path)
    v = d['v_ego'].astype(np.float64)
    radar = d['radar']
    ok = radar[:, 0] > 0.5
    model_ok = d['log_lead_prob'] > 0.5
    lead_x = np.where(ok, radar[:, 1], np.where(model_ok, d['log_lead_x'], np.nan))
    lead_v = np.where(ok, radar[:, 2], np.where(model_ok, d['log_lead_v'], np.nan))
    k2, k3, k4, k6 = 2 * HZ, 3 * HZ, 4 * HZ, 6 * HZ
    v3max = np.array([v[i:i + k3].max() for i in range(len(v))])
    # real gap at 4 s: lead x at t + 4 s over v at t + 4 s
    real_gap4 = shift(lead_x, k4) / np.maximum(shift(v, k4), 1.0)
    rows['x'].append(d['x_plan_feat'])
    rows['plan'].append(d['plan'])
    rows['v'].append(v)
    rows['accel'].append(d['accel'])
    rows['lead_x'].append(lead_x)
    rows['lead_v'].append(lead_v)
    rows['route'].append(np.full(len(v), os.path.basename(path).split('_')[0]))
    rows['real_dv2'].append(shift(v, k2) - v)
    rows['real_dv6'].append(shift(v, k6) - v)
    rows['v3max'].append(v3max)
    rows['real_gap4'].append(real_gap4)
  return {k: np.concatenate(v) for k, v in rows.items()}


def situations(D):
  v, lx, lv = D['v'], D['lead_x'], D['lead_v']
  has_lead = np.isfinite(lx)
  launch = (v > 0.3) & (v < 3) & (D['v3max'] > 3) & np.isfinite(D['real_dv2'])
  follow = has_lead & (lx < 60) & (v > 5) & (np.abs(lv - v) < 2) & np.isfinite(D['real_gap4'])
  cruise = (~has_lead | (lx > 80)) & (v > 8) & np.isfinite(D['real_dv6'])
  return dict(launch=launch, follow=follow, cruise=cruise)


def plan_targets(plan, v, lead_x, lead_v):
  """Model side style numbers for any frames. Bigger = more assertive for all three."""
  vx, px = plan[:, :, COL_VX], plan[:, :, COL_X]
  dv2 = at(vx, 2.0) - vx[:, 0]
  gap4 = (np.nan_to_num(lead_x, nan=200.0) + np.nan_to_num(lead_v, nan=0.0) * 4.0 - at(px, 4.0)) / np.maximum(at(vx, 4.0), 1.0)
  cruise6 = at(vx, 6.0) - v
  return dict(launch=dv2, follow=-gap4, cruise=cruise6)


def controls(D, m):
  v, a = D['v'][m], D['accel'][m]
  lx, lv = D['lead_x'][m], D['lead_v'][m]
  has = np.isfinite(lx).astype(np.float64)
  lx0, lv0 = np.nan_to_num(lx, nan=0.0), np.nan_to_num(lv, nan=0.0)
  return np.column_stack([np.ones_like(v), v, v ** 2, a, a ** 2, a * v, has, lx0, lv0, lv0 - v * has, lx0 / np.maximum(v, 1.0)])


def residual(C, y):
  beta, *_ = np.linalg.lstsq(C, y, rcond=None)
  return y - C @ beta


def ridge(X, y, lam=RIDGE):
  return np.linalg.solve(X.T @ X + lam * len(X) * np.eye(X.shape[1]) / 100.0, X.T @ y)


def loro_r2(X, y, routes):
  """Leave-one-route-out R^2 of a ridge probe, pooled over held-out routes."""
  pred = np.zeros_like(y)
  for r in np.unique(routes):
    test = routes == r
    w = ridge(X[~test], y[~test])
    pred[test] = X[test] @ w
  return 1 - np.mean((y - pred) ** 2) / np.var(y), pred


def main():
  os.makedirs(OUT, exist_ok=True)
  D = load_all()
  S = situations(D)
  x_mean = D['x'].mean(0)
  x_std = D['x'].std(0) + 1e-6
  print(f'dataset: {len(D["v"])} frames, 111 base segments, 11 routes ({", ".join(np.unique(D["route"]))})')

  probes = {}
  for name, m in S.items():
    idx = np.where(m)[0]
    pt = plan_targets(D['plan'][idx], D['v'][idx], D['lead_x'][idx], D['lead_v'][idx])[name]
    real = {'launch': D['real_dv2'], 'follow': -D['real_gap4'], 'cruise': D['real_dv6']}[name][idx]
    C = controls(D, idx)
    X = (D['x'][idx] - x_mean) / x_std
    Xr = X - C @ np.linalg.lstsq(C, X, rcond=None)[0]
    routes = D['route'][idx]
    n_routes = len(np.unique(routes))
    print(f'\n== {name}: {len(idx)} frames, {n_routes} routes')
    for tname, y in (('plan', pt), ('real', real), ('real-plan', real - pt)):
      ok = np.isfinite(y)
      yr = residual(C[ok], y[ok])
      r2_ctrl = 1 - np.var(yr) / np.var(y[ok])
      r2, _ = loro_r2(Xr[ok], yr, routes[ok])
      # route share: how much of the residual is a per-route offset (a driver/car trait would show here)
      route_means = {r: yr[routes[ok] == r].mean() for r in np.unique(routes[ok])}
      route_share = np.var([route_means[r] for r in routes[ok]]) / np.var(yr)
      print(f'  {tname:9s}: controls R2 {r2_ctrl:.2f} | probe on residual LORO R2 {r2:+.2f} | per-route offset share {route_share:.2f}'
            f' | resid std {yr.std():.3f}')
      if tname == 'plan':
        w = ridge(Xr[ok], yr)
        probes[name] = w / np.linalg.norm(w)

  # transfer: does each probe direction predict the other situations' plan residual?
  print('\n== direction cosines (standardized x space)')
  names = list(probes)
  for i, a in enumerate(names):
    print('  ' + a.ljust(7) + ' '.join(f'{probes[a] @ probes[b]:+.2f}' for b in names))
  print('\n== transfer: corr(projection on probe A, plan residual of situation B), all on residualized x')
  for name, m in S.items():
    idx = np.where(m)[0]
    pt = plan_targets(D['plan'][idx], D['v'][idx], D['lead_x'][idx], D['lead_v'][idx])[name]
    C = controls(D, idx)
    X = (D['x'][idx] - x_mean) / x_std
    Xr = X - C @ np.linalg.lstsq(C, X, rcond=None)[0]
    yr = residual(C, pt)
    print(f'  target {name:7s}: ' + ' '.join(f'{a}={np.corrcoef(Xr @ probes[a], yr)[0, 1]:+.2f}' for a in names))
  np.savez(f'{OUT}/probes.npz', x_mean=x_mean, x_std=x_std, **{f'dir_{k}': v for k, v in probes.items()})
  print(f'\nsaved {OUT}/probes.npz')


if __name__ == '__main__':
  main()
