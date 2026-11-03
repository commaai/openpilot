#!/usr/bin/env python3
"""Low/mid-speed corner-cut circuit in the dead plan-head units that 'cut' owns (slots.OWN['cut']).
It replaces lat_readout_wlo20: the base is the master plan, and it is off at highway speed and on straights.

Layer 1 (r4 slots on x_plan_feat), each one a clamp relu(z) - relu(z - c) built from two units:
  m_L, m_R          size of the correction (rank-1 probe of the lateral target), one per sign, capped at CAP
  g_conf            lane confidence probe (lane line logit), ramp 0..1 (optional, off by default)
  g_move, g_slow    speed probe: ramp up at walking speed, ramp down before highway speed
  g_left, g_right   lateral accel probe, ramp 0..1 per turn direction
  g_nolc            no lane change (probe of the lane change prob), ramp 1..0
Layer 2 (r6): AND gate per side, out = relu(m + M * (g_conf + g_move + g_slow + g_nolc + g_turn - 5)), so 0 <= out <= CAP.
  Left turn: the plan cuts when it is left of center, out_L pushes right. Right turn: the opposite.
Output: trunk 11 = out_L, trunk 33 = out_R. Their readout into the y / yaw / yaw rate rows is a least squares
fit to the master plan pulled ALPHA of the way to the lane center.

Probes and readout are fit leave-one-route-out to score; the patch uses all 11 routes.

usage: exp_cut.py [--save [--candidate]] [--set k=v ...]   score held out; --save also writes patches/cut_v1.npz and
       seg/cutC_*.npz (with --candidate: only exp/cut/cut_v1_candidate.npz)
"""
import argparse
import json
import os
import numpy as np

from openpilot.tools.weight_lab import head
from openpilot.tools.weight_lab import fit_lat_edit as F
from openpilot.tools.weight_lab.cut import X_IDXS
from openpilot.tools.weight_lab.exp_cut_data import load
from openpilot.tools.weight_lab.exp_cut_twitch import fmt, steer_curv, twitch
from openpilot.tools.weight_lab.lab_paths import CODE, DATA
from openpilot.tools.weight_lab.slots import Circuit

PATCH = f'{CODE}/patches/cut_v1.npz'
TAG = 'cutC'
LAT_ROWS = np.array([t * 15 + c for c in F.LAT_CH for t in range(33)])
SCALE = head.master_weights()['scale_layer.plan.scale'][LAT_ROWS]
LAM = 1e-2              # probe ridge, on standardized x
OFF_MAX = 2.5           # fits use |offset| below this (no lane changes)
BANDS = ((3, 10), (10, 20), (20, 45))  # speed bands for the inside offset, m/s
HOLD_T = 2.0           # s, the target shift is held constant after this time

CFG = dict(
  alpha=1.0,            # pull the plan this part of the way to the lane center
  gain=1.0,
  cap=0.6,              # max size of the correction (u units: m of push at 1.4 s)
  M=0.6,                # AND gate strength (>= cap: one closed gate shuts the output)
  conf=(1.0, 2.5),      # lane logit probe: gate 0 at the first value, 1 at the second
  move=(5.0, 7.0),      # speed probe, m/s: gate 0 below the first value, 1 above the second
  slow=(17.0, 14.0),    # speed probe, m/s: gate 0 above the first value, 1 below the second
  turn=(0.4, 0.7),      # lateral accel probe, m/s^2
  nolc=(0.3, 0.1),      # lane change prob probe: gate 1 below the second value
  w_y=3.0,              # rank-1 fit: weight of the y rows against the yaw and yaw rate rows
  fit_v=(4.0, 18.0),    # speed range of the correction target and of the rank-1 fit
  teacher=0,            # 1: target is the change of the shipped lat_readout_wlo20 (inside the speed window)
  use_conf=0,           # 1: AND with the lane confidence gate (off: lanes are rarely confident in slow turns)
)


def interp_grid(x, grid_vals):
  """Per-frame np.interp(x[n, :], X_IDXS, grid_vals[n]) for a (n, k) array of x."""
  i = np.clip(np.searchsorted(X_IDXS, x) - 1, 0, len(X_IDXS) - 2)
  w = np.clip((x - X_IDXS[i]) / (X_IDXS[i + 1] - X_IDXS[i]), 0, 1)
  n = np.arange(len(x))[:, None]
  return grid_vals[n, i] * (1 - w) + grid_vals[n, i + 1] * w


def y_at(plan, x=20.0):
  px, py = plan[:, :, 0], plan[:, :, 1]
  i = np.clip((px < x).sum(1) - 1, 0, 31)
  n = np.arange(len(px))
  w = np.clip((x - px[n, i]) / np.maximum(px[n, i + 1] - px[n, i], 1e-6), 0, 1)
  return py[n, i] * (1 - w) + py[n, i + 1] * w


def lat_target(plan, center, alpha):
  """Vectorized fit_lat_edit.target_lat minus the plan: (n, 99) change of y, yaw, yaw rate rows."""
  x, y = plan[:, :, 0].astype(np.float64), plan[:, :, 1].astype(np.float64)
  off = y - interp_grid(np.minimum(x, F.MAX_X), center)
  # far ahead the lanes are not good: past HOLD_T keep the shift of HOLD_T, so the far path only moves in parallel
  off[:, F.T > HOLD_T] = off[:, [np.flatnonzero(F.T <= HOLD_T)[-1]]]
  y2 = y - alpha * np.clip(F.T / F.RAMP_T, 0, 1) * off
  gx = np.maximum(np.gradient(x, axis=1), 1e-3)
  dyaw = np.arctan2(np.gradient(y2, axis=1), gx) - np.arctan2(np.gradient(y, axis=1), gx)
  return np.concatenate([y2 - y, dyaw, np.gradient(dyaw, F.T, axis=1)], axis=1)


def frame_info(D):
  v, prob = D['v'], D['lane_prob']
  c20 = interp_grid(np.full((len(v), 1), 20.0), D['center'])[:, 0]
  return dict(v=v, prob=prob, la=D['curv'] * v ** 2, off=y_at(D['plan']) - c20, c20=c20,
              logit=np.clip(np.log(prob + 1e-9) - np.log(1 - prob + 1e-9), -8, 8))


class Stats:
  """Per-route sums for ridge probes (y may have many columns), so each held-out fold costs one solve."""

  def __init__(self, X, routes, use, y):
    self.routes = sorted(set(routes))
    self.s = {}
    for r in self.routes:
      m = (routes == r) & use
      Xr, yr = X[m], y[m]
      self.s[r] = (len(Xr), Xr.sum(0), Xr.T @ Xr, Xr.T @ yr, yr.sum(0), (yr ** 2).sum(0))

  def sums(self, skip=None):
    return [sum(self.s[r][k] for r in self.routes if r != skip) for k in range(6)]

  def fit(self, skip=None, coef_only=False):
    n, sx, sxx, sxy, sy, _ = self.sums(skip)
    m, ym = sx / n, sy / n
    cov = sxx / n - np.outer(m, m)
    sd = np.sqrt(np.maximum(np.diag(cov), 0)) + 1e-6
    sd_y = sd if np.ndim(ym) == 0 else sd[:, None]
    coef = np.linalg.solve(cov / np.outer(sd, sd) + LAM * np.eye(len(m)), (sxy / n - np.multiply.outer(m, ym)) / sd_y)
    w = coef / sd_y
    if coef_only:
      return w, ym - m @ w, coef, cov / np.outer(sd, sd)
    return w, ym - m @ w


def rank1(S, w_y, skip=None):
  """Best rank-1 linear map x -> u * c for the 99-d lateral target (reduced rank ridge, rows whitened,
  y rows weighted by w_y against the yaw and yaw rate rows).
  Returns the u probe (w, b) and the shape c, scaled so c is a 1 m push to the left at t = T[12] (1.4 s)."""
  w, b, coef, A = S.fit(skip, coef_only=True)
  n, _, _, _, sy, syy = S.sums(skip)
  var = syy / n - (sy / n) ** 2
  wr = np.where(var > 1e-10, 1 / np.sqrt(np.maximum(var, 1e-10)), 0.0)  # whiten rows; rows with no target get no weight
  wr[:33] *= np.sqrt(w_y)
  _, vecs = np.linalg.eigh((coef * wr).T @ A @ (coef * wr))
  c = np.where(wr > 0, vecs[:, -1] / np.where(wr > 0, wr, 1), 0.0)
  c = c / c[12]
  proj = (c * wr ** 2) / ((c * wr ** 2) @ c)
  return (w @ proj, b @ proj), c


def target_mask(D, I, cfg):
  v = I['v']
  m = (v > cfg['fit_v'][0]) & (v < cfg['fit_v'][1]) & (D['lc_prob'] < 0.1)
  return m if cfg['teacher'] else m & (I['prob'] > 0.5)


def lat_targets(D, I, cfg):
  """(n, 99) target change of the lateral rows: the shipped lat_readout_wlo20 change (teacher), or the pull to the lane center."""
  ok = target_mask(D, I, cfg)
  Y = np.zeros((len(ok), 99))
  if cfg['teacher']:
    Y[ok] = np.concatenate([(D['plan_ship'] - D['plan'])[ok][:, :, c] for c in F.LAT_CH], axis=1)
  else:
    Y[ok] = lat_target(D['plan'][ok], D['center'][ok], cfg['alpha'])
  return Y


def build_stats(D, I, Y, cfg):
  X = D['x'].astype(np.float64)
  R = D['route']
  allf = np.ones(len(X), bool)
  fit_lat = target_mask(D, I, cfg) & ((np.abs(I['off']) < OFF_MAX) | (cfg['teacher'] == 1))
  return {
    'lat': Stats(X, R, fit_lat, Y),
    # the gate only matters where the other gates are open: fit lane confidence on curves below highway speed
    'conf': Stats(X, R, (I['v'] > 3) & (I['v'] < 22) & (np.abs(I['la']) > 0.3), np.clip(I['logit'], -3, 4)),
    'speed': Stats(X, R, allf, I['v']),
    'turn': Stats(X, R, I['v'] > 3, I['la']),
    'lc': Stats(X, R, allf, D['lc_prob'].astype(np.float64)),
  }


def layer1(probes, cfg):
  """r4 units per name: a clamp is (unit, same unit with the bias lowered by the top)."""
  units = {}

  def ramp(name, w, b, lo, hi):
    k = 1.0 / (hi - lo)
    units[name] = [(w * k, (b - lo) * k), (w * k, (b - lo) * k - 1.0)]

  # m_L: left turn, the plan is left of center, so the push is to the right (u < 0)
  w_u, b_u = probes['u']
  units['m_L'] = [(-w_u, -b_u), (-w_u, -b_u - cfg['cap'])]
  units['m_R'] = [(w_u, b_u), (w_u, b_u - cfg['cap'])]
  if cfg['use_conf']:
    ramp('g_conf', *probes['conf'], *cfg['conf'])
  ramp('g_move', *probes['speed'], *cfg['move'])
  w_v, b_v = probes['speed']
  ramp('g_slow', -w_v, -b_v, -cfg['slow'][0], -cfg['slow'][1])
  ramp('g_left', *probes['turn'], *cfg['turn'])
  w_t, b_t = probes['turn']
  ramp('g_right', -w_t, -b_t, *cfg['turn'])
  w_l, b_l = probes['lc']
  ramp('g_nolc', -w_l, -b_l, -cfg['nolc'][0], -cfg['nolc'][1])
  return units


def shared_gates(cfg):
  return ('g_conf', 'g_move', 'g_slow', 'g_nolc') if cfg['use_conf'] else ('g_move', 'g_slow', 'g_nolc')


def gates(units, X):
  """Clamped layer-1 value per name."""
  return {name: np.maximum(X @ wa + ba, 0) - np.maximum(X @ wb + bb, 0) for name, ((wa, ba), (wb, bb)) in units.items()}


def and_out(val, cfg):
  names = shared_gates(cfg)
  shared = sum(val[g] for g in names)
  k = len(names) + 1
  out_l = np.maximum(val['m_L'] + cfg['M'] * (shared + val['g_left'] - k), 0)
  out_r = np.maximum(val['m_R'] + cfg['M'] * (shared + val['g_right'] - k), 0)
  return np.stack([out_l, out_r], 1)


def apply_lat(plan, dlat):
  out = plan.copy()
  for k, c in enumerate(F.LAT_CH):
    out[:, :, c] += dlat[:, k * 33:(k + 1) * 33]
  return out


def fit_fold(D, I, S, Y, cfg, skip=None):
  probes = {k: s.fit(skip) for k, s in S.items() if k != 'lat'}
  probes['u'] = rank1(S['lat'], cfg['w_y'], skip)[0]
  units = layer1(probes, cfg)
  tr = (D['route'] != skip) & (I['v'] > 3)
  U = and_out(gates(units, D['x'][tr].astype(np.float64)), cfg)
  # least squares readout (2, 99) in physical plan units, small ridge
  C = np.linalg.solve(U.T @ U + 1e-6 * len(U) * np.eye(2), U.T @ Y[tr]) * cfg['gain']
  C[:, SCALE == 0] = 0.0  # y@t=0 and yaw@t=0 have scale 0: the head can not change them
  return probes, units, C


def q(a):
  return f'p99.9 {np.percentile(a, 99.9):.4f} max {a.max():.4f}' if len(a) else 'n=0'


def score(D, I, plan_new, label, U=None):
  """Inside offset by speed band and steering change vs master."""
  v, prob, la = I['v'], I['prob'], I['la']
  sign = np.sign(D['curv'])
  ins0 = sign * (y_at(D['plan']) - I['c20'])
  ins1 = sign * (y_at(plan_new) - I['c20'])
  dk = np.abs(steer_curv(plan_new, v) - steer_curv(D['plan'], v))
  dyaw = np.abs(plan_new[:, :, 11] - D['plan'][:, :, 11]).max(1)
  curves = (prob > 0.5) & (np.abs(la) > 0.7) & (v > 3)
  print(f'{label}')
  for a, b in BANDS:
    m = curves & (v >= a) & (v < b)
    print(f'  confident curves v {a:2d}-{b:2d} (n={m.sum():4d}): inside@20m median {np.median(ins0[m]):+.3f} -> {np.median(ins1[m]):+.3f}, '
          f'p90 {np.percentile(ins0[m], 90):+.3f} -> {np.percentile(ins1[m], 90):+.3f}, wide (< -0.3 m) {100 * np.mean(ins0[m] < -0.3):.1f}% -> {100 * np.mean(ins1[m] < -0.3):.1f}%')
  lo = (prob < 0.5) & (v > 3)
  lo3 = lo & (prob < 0.3)
  crawl = v < 3
  straight = (np.abs(la) < 0.3) & (v > 3)
  lc = D['lc_prob'] > 0.1
  fast = (v > 20) & (v < 45)  # v ~80 m/s frames are bad speed readings
  print(f'  |d curvature| vs master: confident curves {q(dk[curves])} | NOT confident (v>3) {q(dk[lo])}, lane prob < 0.3 {q(dk[lo3])}')
  print(f'  v<3: {q(dk[crawl])}, max |d yaw| {dyaw[crawl].max():.4f} rad | straight (|lat acc|<0.3, v>3) {q(dk[straight])} | '
        f'lane change {q(dk[lc])} | v>20 {q(dk[fast])}')
  if U is not None:
    on = U.sum(1) > 1e-6
    print(f'  circuit on: {100 * on[curves & (v < 20)].mean():.1f}% of confident curves below 20 m/s, {100 * on[fast].mean():.2f}% of v>20, '
          f'{100 * on[lo].mean():.2f}% not confident, {100 * on[crawl].mean():.2f}% v<3, {100 * on[straight].mean():.2f}% straight')
  print(f'  twitch (v>20): {fmt(twitch(D, plan_new))}')
  t = np.array([60 * int(sg.rsplit('_', 1)[1]) for sg in D['seg']]) + D['frame'] / 20
  prefix = np.array([sg.rsplit('_', 1)[0] for sg in D['seg']])
  print('  clips, mean |inside@20m| on confident frames (median v):')
  for c in json.load(open(f'{CODE}/clips.json')):
    m = (prefix == c['prefix']) & (t >= c['start']) & (t <= c['end']) & (prob > 0.5)
    if m.sum() >= 5:
      print(f"    {c['prefix']:>4} {c['start']:4d}-{c['end']:4d} s {c['kind']:7s} n={m.sum():3d} v {np.median(v[m]):4.1f}: "
            f"{np.abs(ins0[m]).mean():.2f} -> {np.abs(ins1[m]).mean():.2f}  {c['label']}")


def build_circuit(units, C, cfg):
  """Write the fitted circuit into the cut units."""
  circ = Circuit('cut')
  slot = {}
  r4_free = iter(circ.free('r4'))
  for name, rows in units.items():
    slot[name] = [circ.r4(next(r4_free), w, b) for w, b in rows]
  for side, turn, trunk_unit, c in (('L', 'g_left', 11, C[0]), ('R', 'g_right', 33, C[1])):
    w = {}
    m_a, m_b = slot[f'm_{side}']
    w[m_a], w[m_b] = 1.0, -1.0
    for g in shared_gates(cfg) + (turn,):
      g_a, g_b = slot[g]
      w[g_a], w[g_b] = cfg['M'], -cfg['M']
    k = circ.r6(circ.free('r6')[0], w, -(len(shared_gates(cfg)) + 1) * cfg['M'])
    readout = np.zeros(990)
    ok = SCALE > 0
    readout[LAT_ROWS[ok]] = c[ok] / SCALE[ok]
    circ.out(trunk_unit, {k: 1.0}, readout)
  return circ


def recompute_curv(d, plan_new):
  """Shift the replayed curv by the change in plan curvature; hold it below the min lat control speed like replay.py."""
  dk = steer_curv(plan_new, d['v_ego']) - steer_curv(d['plan'], d['v_ego'])
  out = d['curv'].copy()
  prev = 0.0
  for i in range(len(out)):
    if d['v_ego'][i] > 0.3:
      prev = dk[i]
    out[i] += prev
  return out


def save_all(D, I, S, Y, cfg, candidate):
  _, units, C = fit_fold(D, I, S, Y, cfg)
  circ = build_circuit(units, C, cfg)
  tmp = f'{DATA}/exp/cut/cut_v1_candidate.npz'
  circ.save(tmp, f'low/mid-speed gated corner cut (replaces lat_readout_wlo20), {cfg}')
  master = head.master_weights()
  patched = head.patched_weights([tmp])
  # check: the numpy circuit equals the patched head
  x = D['x'][::20].astype(np.float64)
  dplan = head.forward(patched, x)['plan'] - head.forward(master, x)['plan']
  ref = apply_lat(np.zeros_like(dplan), and_out(gates(units, x), cfg) @ C)
  print(f'patched head vs numpy circuit: max diff {np.abs(dplan - ref).max():.2e}, max change {np.abs(ref).max():.3f}')
  assert np.abs(dplan - ref).max() < 1e-4
  others = [p for p in head.SHIPPED if 'lat_readout' not in p]
  head.patched_weights(others + [tmp])  # raises on a clash with stop_gain / soft_brake
  if candidate:
    print(f'kept the candidate in {tmp}, nothing written to patches/ or seg/')
    return
  os.replace(tmp, PATCH)
  print(f'saved {PATCH}')
  for p in head.base_files():
    d = dict(np.load(p))
    d['plan'] = head.edited_plan(d, patched, head.forward(master, d['x_plan_feat']))
    d['curv'] = recompute_curv(np.load(p), d['plan'])
    np.savez_compressed(f'{DATA}/seg/{TAG}_{os.path.basename(p)}', **d)
  print(f'wrote {len(head.base_files())} seg/{TAG}_* replays (master + cut_v1 only)')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--save', action='store_true')
  ap.add_argument('--candidate', action='store_true', help='with --save: write only exp/cut/cut_v1_candidate.npz')
  ap.add_argument('--set', nargs='*', default=[], help='override CFG, e.g. alpha=0.8 conf=2,3')
  ap.add_argument('--shipped', action='store_true', help='also score lat_readout_wlo20 alone')
  args = ap.parse_args()
  cfg = dict(CFG)
  for kv in args.set:
    k, val = kv.split('=')
    cfg[k] = tuple(float(a) for a in val.split(',')) if ',' in val else float(val)
  print(cfg)
  D = load()
  I = frame_info(D)
  if args.shipped:
    score(D, I, D['plan_ship'], 'lat_readout_wlo20 vs master (fit on all routes, NOT held out)')
  Y = lat_targets(D, I, cfg)
  ok = target_mask(D, I, cfg)
  S = build_stats(D, I, Y, cfg)
  plan_new = D['plan'].copy()
  U_all = np.zeros((len(ok), 2))
  for r in sorted(set(D['route'])):
    _, units, C = fit_fold(D, I, S, Y, cfg, skip=r)
    te = D['route'] == r
    U = and_out(gates(units, D['x'][te].astype(np.float64)), cfg)
    U_all[te] = U
    plan_new[te] = apply_lat(D['plan'][te], U @ C)
  score(D, I, plan_new, 'cut circuit vs master, HELD OUT by route', U_all)
  os.makedirs(f'{DATA}/exp/cut', exist_ok=True)
  np.save(f'{DATA}/exp/cut/heldout_plan.npy', plan_new)
  if args.save:
    save_all(D, I, S, Y, cfg, args.candidate)


if __name__ == '__main__':
  main()
