#!/usr/bin/env python3
"""Gated, capped corner-cut circuit written into four dead plan-trunk units (no change anywhere else).

A dead unit j is rewired so that h_j = relu(w . x + b), x = 512-d temporal feature into the plan head
(in_layer row j set, its links into and out of the residual block zeroed). Per sign s in (+1, -1):
  z_s = s * off(x) + M * (gate(x) - THETA) - TAU
  out_s = relu(z_s) - relu(z_s - CAP)          (main unit minus cap unit: 0 <= out_s <= CAP)
off  = ridge probe of the plan's offset from lane center @20 m (confident-lane frames, v > 8 m/s)
gate = ridge probe of "confident lanes and v > 8 m/s and no lane change" (0/1), false positives weighted up.
When gate is low both outputs stay at 0, and the cap bounds the change on any frame, in or out of distribution.
Readout columns (lateral rows y, yaw, yaw rate) are fit by least squares to the corner-cut correction.

Scored leave-one-route-out; writes seg/cut<tag>_<seg>.npz and cut<tag>_circuit.npz.

usage: fit_lat_gated.py TAG [M] [TAU] [GAIN] [CAP] [THETA]
"""
import glob
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_lat_edit as F  # noqa: E402
from cut import inside_offset  # noqa: E402

HERE = F.HERE
UNITS = (11, 33)       # main units for s = +1, -1
CAP_UNITS = (47, 48)   # cap units for s = +1, -1
FP_WEIGHT = 3.0


def ridge_raw(X, y, lam, w=None):
  """Weighted ridge on standardized X, returned as raw-space weights and bias."""
  w = np.ones(len(X)) if w is None else w
  m, sd = np.average(X, 0, w), X.std(0) + 1e-6
  Z = (X - m) / sd
  ym = np.average(y, weights=w)
  Zw = Z * w[:, None]
  coef = np.linalg.solve(Zw.T @ Z + lam * w.sum() * np.eye(Z.shape[1]), Zw.T @ (y - ym))
  w_raw = coef / sd
  return w_raw, ym - m @ w_raw


def load_all(paths):
  out = []
  for p in paths:
    d = np.load(p)
    _, prob = F.lane_center_all(d['raw'])
    _, raw_off, _, _ = inside_offset(d)
    gate = (prob > 0.5) & (d['v_ego'] > 8) & (F.lane_change_prob(d['raw']) < 0.1)
    out.append(dict(path=p, route=str(d['seg'][0]).rsplit('/', 1)[0], x=d['x_plan_feat'], off=raw_off[:, 1],
                    gate=gate, ok=gate & (np.abs(raw_off[:, 1]) < 2.5), v=d['v_ego']))
  return out


def unit_out(x, circ):
  z = x @ circ['w_in'].T + circ['b_in']
  return np.maximum(z, 0) - np.maximum(z - circ['cap'], 0)


def fit_circuit(parts, dys, M, tau, gain, cap, theta, lam=1e-2):
  X = np.concatenate([p['x'][p['ok']] for p in parts]); y = np.concatenate([p['off'][p['ok']] for p in parts])
  w_o, b_o = ridge_raw(X, y, lam)
  Xa = np.concatenate([p['x'] for p in parts]); g = np.concatenate([p['gate'] for p in parts]).astype(np.float64)
  w_g, b_g = ridge_raw(Xa, g, lam, np.where(g > 0.5, 1.0, FP_WEIGHT))
  circ = dict(w_in=np.stack([w_o + M * w_g, -w_o + M * w_g]), b_in=np.array([b_o + M * (b_g - theta) - tau, -b_o + M * (b_g - theta) - tau]), cap=cap)
  U = np.concatenate([unit_out(p['x'][p['v'] > 3], circ) for p in parts])
  Y = np.concatenate(dys)
  C = np.linalg.solve(U.T @ U + 1e-3 * np.eye(2) * len(U), U.T @ Y)  # (2, 99) in weight units
  circ['C'] = C * gain
  return circ


def apply(d, circ, scale):
  U = unit_out(d['x_plan_feat'], circ)
  dmu = (U @ circ['C']) * scale
  plan = d['plan'].copy()
  for k, c in enumerate(F.LAT_CH):
    plan[:, :, c] += dmu[:, k * 33:(k + 1) * 33]
  return plan, U


def main():
  tag = sys.argv[1]
  M, tau, gain, cap, theta = [float(a) for a in (sys.argv[2:7] + ['4', '0.1', '1.0', '1.0', '1.0'][len(sys.argv) - 2:])]
  lat, scale = F.load(1.0)  # correction targets (dy) per segment, on frames with v > 3
  parts = load_all([p['path'] for p in lat])  # same files as the targets, even if new replays land meanwhile
  routes = sorted({p['route'] for p in parts})
  print(f'gated circuit M {M} theta {theta} tau {tau} gain {gain} cap {cap}: {len(parts)} segments, {len(routes)} routes')
  pool = {'in0': [], 'in1': [], 'dk_lo': [], 'dk_hi': [], 'dk_slow': [], 'leak': []}
  for r in routes:
    tr = [i for i, p in enumerate(parts) if p['route'] != r]
    circ = fit_circuit([parts[i] for i in tr], [lat[i]['dy'] for i in tr], M, tau, gain, cap, theta)
    for i in [i for i, p in enumerate(parts) if p['route'] == r]:
      base = np.load(parts[i]['path'])
      d = dict(base)
      plan, U = apply(d, circ, scale)
      d['plan'] = plan.astype(np.float32)
      ins0, ins1 = inside_offset(base), inside_offset(d)
      la = np.abs(d['curv'] * d['v_ego'] ** 2)
      m = (ins0[2] > 0.5) & (d['v_ego'] > 8) & (la > 0.7)
      lo = (ins0[2] < 0.5) & (d['v_ego'] > 3)
      dk = np.abs(F.steer_curv(d['plan'], d['v_ego']) - F.steer_curv(base['plan'], d['v_ego']))
      pool['in0'].append(ins0[0][m, 1]); pool['in1'].append(ins1[0][m, 1])
      pool['dk_lo'].append(dk[lo]); pool['dk_hi'].append(dk[~lo]); pool['dk_slow'].append(dk[d['v_ego'] < 8])
      pool.setdefault('dk_lo_fast', []).append(dk[lo & (d['v_ego'] > 8)])
      pool.setdefault('on_slow', []).append(U[d['v_ego'] < 3].sum(1) > 0)
      pool['leak'].append(U[lo].sum(1) > 0)
      np.savez_compressed(f"{HERE}/seg/cut{tag}_{os.path.basename(parts[i]['path'])[:-4]}.npz", **d)
  P = {k: np.concatenate(v) for k, v in pool.items()}
  q = lambda a: f'p95 {np.percentile(a, 95):.4f} p99.9 {np.percentile(a, 99.9):.4f} max {a.max():.4f}'
  print(f'  HELD OUT, confident-lane curves (n={len(P["in0"])}): inside@20m median {np.median(P["in0"]):+.2f} -> {np.median(P["in1"]):+.2f}, '
        f'p90 {np.percentile(P["in0"], 90):+.2f} -> {np.percentile(P["in1"], 90):+.2f}, mean |inside| {np.abs(P["in0"]).mean():.2f} -> {np.abs(P["in1"]).mean():.2f}')
  print(f'  steering change |d curvature| confident: {q(P["dk_hi"])} | NOT confident: {q(P["dk_lo"])} | v < 8 m/s: {q(P["dk_slow"])}')
  print(f'  AT SPEED (v > 8) not-confident steering change: {q(P["dk_lo_fast"])}')
  print(f'  circuit active on {100 * P["leak"].mean():.1f}% of not-confident frames (v > 3), {100 * P["on_slow"].mean():.2f}% of frames below 3 m/s')
  full = fit_circuit(parts, [p['dy'] for p in lat], M, tau, gain, cap, theta)
  np.savez(f'{HERE}/cut{tag}_circuit.npz', units=np.array(UNITS), cap_units=np.array(CAP_UNITS), scale=scale, M=M, tau=tau, theta=theta, **full)


if __name__ == '__main__':
  main()
