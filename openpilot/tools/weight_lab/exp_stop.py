#!/usr/bin/env python3
"""STOP-line full stop: a gated circuit in the 'stop' units that asks the plan to stop at painted STOP lines.

Problem: at STOP lines the plan never asks for v = 0, it rolls the stop. Red lights are already fine.

Circuit (owner 'stop', see slots.py):
  s = w . x + b                 linear STOP probe on x_plan_feat (ridge, 1 = 0-30 m before a STOP line and moving,
                                0 = everything else, red lights and greens included)
  layer 1: r4 unit 47 = relu(s - t), r4 slot 9 = relu(s - t - cap)   (skip path of slot 9 cancelled by Circuit)
  layer 2: r6 unit = relu(slot 9)
  trunk 47 = relu(relu(s - t) - relu(s - t - cap)) = clip(s - t, 0, cap)
  readout: long rows (x, vx, ax) only, least-squares fit to a kinematic stop at the line on the training STOP frames.
The probe is ~0 when the car is stopped, so the push ends at v ~ 0 and the model's own plan does the resume.

Scoring is leave-one-route-out: for each route, fit on the other 10 routes and score the held-out one.
The saved patch is fit on all routes.

usage: exp_stop.py [--save]      (--save writes patches/stop_v1.npz)
"""
import argparse
import os
import numpy as np

from openpilot.tools.weight_lab import head
from openpilot.tools.weight_lab.lab_paths import CODE, DATA
from openpilot.tools.weight_lab.lights import events
from openpilot.tools.weight_lab.lights_edit import accels, T
from openpilot.tools.weight_lab.slots import Circuit, DEAD_R4, N_PLAN
from openpilot.tools.weight_lab.exp_stop_labels import STOP, RED

OUT = f'{DATA}/exp/stop'
CACHE = f'{OUT}/cache_v1.npz'
CIRCUIT = [69, 248, 201, 239]
GREEN_EPS = [('f7_13', 593, 710), ('f7_17', 219, 311), ('f7_19', 132, 206), ('s608_8', 769, 866), ('fe_4', 428, 485),
             ('fd_6', 898, 972)]  # camera-checked: light ahead, the car did not stop
NEAR = 30.0      # m before the line: positive probe frames
V_POS = 0.3      # m/s: positive probe frames must move faster than this
LAM = 0.03       # ridge strength
CAP = float(os.environ.get('CAP', 0.5))      # gate saturation
PCT = float(os.environ.get('PCT', 99.9))     # gate threshold: this percentile of the probe on training open road
HOLD = float(os.environ.get('HOLD', 1.5))    # s at 0 in the target plan
A_GO = 1.0       # m/s² pull away in the target plan
BOOST = float(os.environ.get('BOOST', 1.0))  # probe target in the last 8 m
NEG_PEN = float(os.environ.get('NEG_PEN', 10.0))  # weight of the penalty on planned speed below V_FLOOR
V_FLOOR = -0.3   # m/s: shipped red-light plans also dip to about -0.1..-0.4
SLOT_B = 9
LONG_ROWS = np.array([t * 15 + c for t in range(33) for c in (0, 3, 6)])


def mark(d, eps, name):
  """Distance to the line for approach frames (inf elsewhere) and the episode id (-1 elsewhere)."""
  v, fr = d['v_ego'], d['frame']
  pos = np.concatenate([[0], np.cumsum(v * 0.05)])[:-1]
  dist = np.full(len(v), np.inf)
  eid = np.full(len(v), -1)
  for k, (n, f0, f1) in enumerate(eps):
    if n != name:
      continue
    w = np.flatnonzero((fr >= f0) & (fr <= f1))
    stopped = w[v[w] < 0.5]
    line = stopped[0] if len(stopped) else w[np.argmin(v[w])]  # first stopped frame, or the slowest frame of a rolling stop
    a = w[w <= line]
    dist[a] = pos[line] - pos[a]
    eid[w] = k
  return dist, eid


def build_cache():
  ship = head.patched_weights(head.SHIPPED)
  master = head.master_weights()
  keys = ('x', 'v', 'route', 'name', 'frame', 'plan0', 'a0', 'circ', 'sd', 'se', 'rd', 're', 'ge', 'ev_d', 'seg_id')
  rows = {k: [] for k in keys}
  for i, p in enumerate(head.base_files()):
    name = os.path.basename(p)[:-4]
    d = np.load(p)
    plan0 = head.edited_plan(d, ship, head.forward(master, d['x_plan_feat']))
    n = len(d['v_ego'])
    sd, se = mark(d, STOP, name)
    rd, re = mark(d, RED, name)
    _, ge = mark(d, GREEN_EPS, name)
    # distance to the next controlled stop of any kind (lights.events) or labeled line, for open-road frames
    ev = [(name, e['onset'], e['stop']) for e in events(d)]
    ev_d, _ = mark(d, ev, name)
    ev_d = np.minimum(ev_d, np.minimum(sd, rd))
    vals = (d['x_plan_feat'].astype(np.float32), d['v_ego'], np.full(n, str(d['seg'][0]).rsplit('/', 1)[0]), np.full(n, name),
            d['frame'], plan0.astype(np.float32), accels(plan0), d['x_plan_trunk'][:, CIRCUIT], sd, se, rd, re, ge, ev_d, np.full(n, i))
    for k, val in zip(keys, vals, strict=True):
      rows[k].append(val)
  os.makedirs(OUT, exist_ok=True)
  np.savez(CACHE, **{k: np.concatenate(v) for k, v in rows.items()})


def far_from_stop(z, gap=250.0):
  """Frames more than GAP m (driven) before any stop / labeled line and not inside a labeled episode."""
  far = np.ones(len(z['v']), bool)
  for s in np.unique(z['seg_id']):
    m = np.flatnonzero(z['seg_id'] == s)
    v = z['v'][m]
    pos = np.concatenate([[0], np.cumsum(v * 0.05)])[:-1]
    lines = m[np.flatnonzero(np.isfinite(z['ev_d'][m]) & (z['ev_d'][m] < 0.5))]
    for li in lines:
      j = li - m[0]
      far[m[(pos[j] - pos < gap) & (np.arange(len(m)) <= j)]] = False
  return far & (z['se'] < 0) & (z['re'] < 0) & (z['ge'] < 0)


def ridge(X, y, lam):
  mu, sd = X.mean(0), X.std(0) + 1e-6
  Z = (X - mu) / sd
  w = np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
  return w / sd, float(y.mean() - (mu / sd) @ w)


def stop_target(v0, dist, plan):
  """Kinematic plan that stops at the line: constant decel, hold 0 for HOLD s, then pull away at A_GO
  (never faster than the model's own plan). Long rows only, (33, 15) like plan."""
  a = np.clip(v0 ** 2 / (2 * max(dist, 1.0)), 0.3, 3.5)
  t_stop = v0 / a
  tf = np.linspace(0, 10, 401)
  go = np.minimum(np.maximum(A_GO * (tf - t_stop - HOLD), 0.0), np.maximum(np.interp(tf, T, plan[:, 3]), 0.0))
  vf = np.where(tf < t_stop, v0 - a * tf, go)
  xf = np.concatenate([[0], np.cumsum((vf[1:] + vf[:-1]) / 2 * np.diff(tf))])
  out = plan.copy()
  out[:, 3] = np.interp(T, tf, vf)
  out[:, 0] = np.interp(T, tf, xf)
  out[:, 6] = np.interp(T, tf[:-1], np.diff(vf) / np.diff(tf))
  return out


def fit(z, train, open_frames):
  """Probe, threshold and readout from the training frames."""
  v, sd, se = z['v'], z['sd'], z['se']
  pos = (sd < NEAR) & (v > V_POS)
  # frames of a STOP episode past the line are negative (the push must end there, for the resume);
  # frames more than NEAR m before the line are left out
  neg = ~pos & ((se < 0) | ~np.isfinite(sd))
  idx = np.flatnonzero(train & (pos | neg))
  idx = idx[pos[idx] | (np.arange(len(idx)) % 2 == 0)]  # every 2nd negative, to keep the solve small
  y = pos[idx] * np.where(sd[idx] < 8, BOOST, 1.0)  # the last 8 m get a larger target: the gate must stay on to the line
  w, b = ridge(z['x'][idx].astype(np.float64), y, LAM)
  s = z['x'] @ w + b
  t = float(np.percentile(s[train & open_frames], PCT))  # (100 - PCT)% of training open-road frames reach the gate
  g = np.clip(s - t, 0, CAP)
  # readout of the speed rows: per time step, least squares g * r = target - plan over training STOP approach frames,
  # plus a penalty when the edited plan speed goes below 0
  fi = np.flatnonzero(train & np.isfinite(sd) & (sd < NEAR) & (v > V_POS) & (g > 0))
  p0 = z['plan0'][fi, :, 3]
  dv = np.array([stop_target(z['plan0'][i, 0, 3], sd[i], z['plan0'][i])[:, 3] for i in fi]) - p0
  wt = np.where(sd[fi] < 10, 2.0, 1.0)  # the last 10 m decide if the car stops
  grid = np.linspace(-30, 5, 3501)
  r_v = np.zeros(33)
  for k in range(1, 33):
    fit_err = ((dv[:, k][None] - grid[:, None] * g[fi][None]) ** 2 * wt).sum(1)
    below = (np.maximum(V_FLOOR - (p0[:, k][None] + grid[:, None] * g[fi][None]), 0) ** 2 * wt).sum(1)
    r_v[k] = grid[np.argmin(fit_err + NEG_PEN * below)]
  # smooth in time (0.3 s), keep 0 change at t = 0; x and ax rows follow from the speed rows, so the plan stays consistent
  tf = np.linspace(0, 10, 1001)
  vf = np.interp(tf, T, r_v)
  ker = np.exp(-0.5 * (np.arange(-90, 91) * 0.01 / 0.3) ** 2)
  vf = np.convolve(np.pad(vf, 90, mode='edge'), ker / ker.sum(), mode='valid')
  vf -= vf[0]
  r_v = np.interp(T, tf, vf)
  r_x = np.interp(T, tf, np.concatenate([[0], np.cumsum((vf[1:] + vf[:-1]) / 2 * np.diff(tf))]))
  r_a = np.interp(T, tf, np.gradient(vf, tf))
  r = np.zeros(495)
  for k in range(33):
    r[k * 15 + 0], r[k * 15 + 3], r[k * 15 + 6] = r_x[k], r_v[k], r_a[k]
  return dict(w=w, b=b, t=t, r=r)


def circuit(params, P_ship):
  """Write the fitted gate and readout into the 'stop' units."""
  c = Circuit('stop')
  w, b, t = params['w'], params['b'], params['t']
  c.r4(SLOT_B, w, b - t - CAP)
  k = c.free('r6')[0]
  c.r6(k, {SLOT_B: 1.0}, 0.0)
  scale = P_ship['scale_layer.plan.scale'][:495]
  readout = np.zeros(N_PLAN)
  readout[LONG_ROWS] = params['r'][LONG_ROWS] / scale[LONG_ROWS]
  c.out(47, {k: -1.0}, readout, w_x=w, b_x=b - t, b=0.0)
  # the skip cancel writes a full row of res_layer.plan.0, also zeros into dead r4 columns that other patches own.
  # patch_model.py clears those columns anyway, so drop these zero writes to stack with the soft brake.
  own_r4 = {u for kind, u in c.used if kind == 'r4'}
  for key in [key for key, val in c.w.items() if key[0] == 'res_layer.plan.0.weight' and val == 0.0 and
              key[1] % 256 in DEAD_R4 and key[1] % 256 not in own_r4]:
    del c.w[key]
  return c


def plans_stop(plans, dist, margin=5.0):
  """True where the plan gets below 0.3 m/s before it is MARGIN m past the line."""
  out = []
  for p, d in zip(plans, dist, strict=True):
    slow = np.flatnonzero(p[:, 3] < 0.3)
    out.append(len(slow) > 0 and p[slow[0], 0] < d + margin)
  return np.array(out, dtype=float)


def score(z, plan1, mask, open_frames):
  """Held-out numbers on the frames in MASK. plan1: edited plan for every frame."""
  v, sd, rd = z['v'], z['sd'], z['rd']
  a1 = np.zeros(len(v))
  for s in np.unique(z['seg_id'][mask]):
    m = z['seg_id'] == s
    a1[m] = accels(plan1[m])
  a0 = z['a0']
  da = a1 - a0
  res = {}
  rows = []
  for k in np.unique(z['se'][mask & (z['se'] >= 0)]):
    m = (z['se'] == k)
    ap = np.flatnonzero(m & np.isfinite(sd) & (sd > 3) & (sd < NEAR) & (v > V_POS))
    if len(ap) == 0:
      continue

    def at_line(plan):
      # planned speed when the plan reaches the line, per approach frame
      return np.array([np.interp(sd[i], plan[i, :, 0], plan[i, :, 3]) if plan[i, -1, 0] > sd[i] else 0.0 for i in ap])
    held = m & (v < 0.3)
    rows.append((z['name'][m][0], int(z['frame'][m][0]), v[m].min(), np.median(at_line(z['plan0'])), np.median(at_line(plan1)),
                 z['plan0'][ap][:, :, 3].min(), plan1[ap][:, :, 3].min(), np.median(z['plan0'][ap, -1, 3]), np.median(plan1[ap, -1, 3]),
                 a0[m].min(), a1[m].min(), float(np.abs(da[held]).max()) if held.any() else np.nan,
                 plans_stop(z['plan0'][ap], sd[ap]).mean(), plans_stop(plan1[ap], sd[ap]).mean()))
  res['stop'] = rows
  red = np.flatnonzero(mask & (z['re'] >= 0))
  res['red'] = []
  for k in np.unique(z['re'][red]):
    m = z['re'] == k
    ap = np.flatnonzero(m & np.isfinite(rd) & (rd > 3) & (rd < NEAR) & (v > V_POS))
    res['red'].append((z['name'][m][0], int(z['frame'][m][0]), a0[m].min(), a1[m].min(), da[m].min(), da[m].max(),
                       plans_stop(z['plan0'][ap], rd[ap]).mean() if len(ap) else np.nan,
                       plans_stop(plan1[ap], rd[ap]).mean() if len(ap) else np.nan))
  green = mask & ((z['ge'] >= 0) | ((z['circ'][:, 3] > 0.2) & (z['circ'][:, :3].sum(1) < 0.05) & (z['se'] < 0) & (z['re'] < 0) &
                                    ~np.isfinite(z['ev_d']) & (v > 3)))
  res['green'] = -da[green]
  op = mask & open_frames & (v > 8)
  res['open'] = da[op]
  res['all_other'] = da[mask & (z['se'] < 0) & (z['re'] < 0)]
  return res, a1


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--save', action='store_true')
  args = ap.parse_args()
  if not os.path.exists(CACHE):
    build_cache()
  z = dict(np.load(CACHE))
  open_frames = far_from_stop(z)
  P_ship = head.patched_weights(head.SHIPPED)
  ship_plan = head.forward(P_ship, z['x'])['plan']
  routes = sorted(set(z['route']))
  plan1 = np.zeros_like(z['plan0'])
  tag = f'pct{PCT}_cap{CAP}_boost{BOOST}_pen{NEG_PEN}'
  tmp = f'{OUT}/fold_{tag}.npz'
  for r in routes:
    test = z['route'] == r
    params = fit(z, ~test, open_frames)
    circuit(params, P_ship).save(tmp, 'fold')
    P = head.patched_weights(head.SHIPPED + [tmp])
    plan1[test] = z['plan0'][test] + (head.forward(P, z['x'][test])['plan'] - ship_plan[test]).astype(np.float32)
    print(f'fold {r}: t {params["t"]:.2f}')
  res, a1 = score(z, plan1, np.ones(len(z['v']), bool), open_frames)
  np.savez(f'{OUT}/loro_{tag}.npz', plan1=plan1, a1=a1)
  s = np.array([row[2:] for row in res['stop']], dtype=np.float64)
  print(f'\nSTOP events, held out by route (n={len(s)}), shipped -> edit: name frame | car vmin | plan v at line | '
        'plan vmin (any horizon) | plan v@10s | accel min | max |da| while stopped')
  for row in res['stop']:
    print(f'  {row[0]:>8} f{row[1]:4d} | {row[2]:.1f} | {row[3]:.2f} -> {row[4]:.2f} | {row[5]:+.2f} -> {row[6]:+.2f} | '
          f'{row[7]:4.1f} -> {row[8]:4.1f} | {row[9]:+.2f} -> {row[10]:+.2f} | {row[11]:.2f}')
  print(f'  median: plan v at line {np.median(s[:, 1]):.2f} -> {np.median(s[:, 2]):.2f} (events < 0.5 m/s: {np.sum(s[:, 1] < 0.5)} -> '
        f'{np.sum(s[:, 2] < 0.5)}) | plan vmin {np.median(s[:, 3]):+.2f} -> {np.median(s[:, 4]):+.2f} (events < 0.3: {np.sum(s[:, 3] < 0.3)} -> '
        f'{np.sum(s[:, 4] < 0.3)}, lowest {s[:, 4].min():+.2f}) | v@10s {np.median(s[:, 5]):.1f} -> {np.median(s[:, 6]):.1f} | '
        f'accel min {np.median(s[:, 7]):+.2f} -> {np.median(s[:, 8]):+.2f}')
  print(f'  approach frames 3-30 m where the plan stops (< 0.3 m/s) within 5 m past the line: mean over events '
        f'{100 * np.mean(s[:, 10]):.0f}% -> {100 * np.mean(s[:, 11]):.0f}%')
  rd = np.array([row[2:] for row in res['red']])
  print(f'\nRED events (n={len(rd)}): accel min median {np.median(rd[:, 0]):+.2f} -> {np.median(rd[:, 1]):+.2f}, '
        f'worst extra brake {rd[:, 2].min():+.2f}, worst extra accel {rd[:, 3].max():+.2f}, '
        f'plan stops within 5 m of the line {100 * np.nanmean(rd[:, 4]):.0f}% -> {100 * np.nanmean(rd[:, 5]):.0f}%')
  for row in res['red']:
    if abs(row[4]) > 0.2 or abs(row[5]) > 0.2:
      print(f'  {row[0]:>8} f{row[1]:4d}: accel min {row[2]:+.2f} -> {row[3]:+.2f}, da {row[4]:+.2f}..{row[5]:+.2f}')
  g, op, oth = res['green'], res['open'], res['all_other']
  print(f'\nGREEN frames (n={len(g)}): extra brake > 0.3 on {100 * np.mean(g > 0.3):.2f}%, max {g.max():.2f}')
  print(f'OPEN road (v > 8, > 250 m from any stop, n={len(op)}): |da| > 0.3 on {100 * np.mean(np.abs(op) > 0.3):.3f}%, '
        f'p99.9 {np.percentile(np.abs(op), 99.9):.3f}, max {np.abs(op).max():.2f}')
  print(f'ALL frames outside STOP/RED episodes (n={len(oth)}): |da| > 0.3 on {100 * np.mean(np.abs(oth) > 0.3):.3f}%, '
        f'p99.9 {np.percentile(np.abs(oth), 99.9):.3f}, max {np.abs(oth).max():.2f}')
  if args.save:
    params = fit(z, np.ones(len(z['v']), bool), open_frames)
    out = f'{CODE}/patches/stop_v1.npz'
    circuit(params, P_ship).save(out, f'STOP-line full stop: ridge probe gate (t {params["t"]:.2f}, cap {CAP}) into trunk 47, '
                                      'kinematic-stop readout on long rows')
    head.patched_weights(head.SHIPPED + [out])  # raises on a clash with the shipped patches


if __name__ == '__main__':
  main()
