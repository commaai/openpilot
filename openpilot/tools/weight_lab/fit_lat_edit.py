#!/usr/bin/env python3
"""Fit a corner-cut correction into the plan head's lateral readout (y, yaw, yaw rate rows), closed form.

Target per frame: the model's own plan, pulled toward its own lane center (confident lanes only) with a
ramp over RAMP_T seconds, with yaw and yaw rate rebuilt from the new path. Frames without confident lanes,
or in a lane change, keep the original plan as target, so the fit learns to leave them alone.
Delta = ridge fit from the 256-d trunk (+ bias) to (target - plan) / scale, kept small by LAM.

Scored leave-one-route-out. Writes, per held-out route, edited copies of the replays (seg/cut<tag>_<prefix>_<n>.npz)
with plan = plan + delta from the fold that did not see that route, and the all-route delta to cut<tag>_delta.npz.

usage: fit_lat_edit.py TAG ALPHA [LAM] [W_LO]
"""
import glob
import os
import sys
import numpy as np

from openpilot.selfdrive.controls.lib.drive_helpers import get_curvature_from_plan

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cut import inside_offset  # noqa: E402

from openpilot.tools.weight_lab.lab_paths import CODE, DATA as HERE  # noqa: F401
H = 'policy_model.temporal_hydra.'
T = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
X_IDXS = np.array([192.0 * (i / 32) ** 2 for i in range(33)])
LAT_CH = (1, 11, 14)  # y, yaw, yaw rate
ROWS = np.array([t * 15 + c for c in LAT_CH for t in range(33)])
RAMP_T = 1.5
MAX_X = 60.0
SKIP = ('g2', 'fdp')


def lane_center_all(raw):
  lanes = raw[:, 117:645][:, :264].reshape(-1, 4, 33, 2)[..., 0]
  prob = 1 / (1 + np.exp(-raw[:, 645:653].reshape(-1, 4, 2)[..., 1]))
  return (lanes[:, 1] + lanes[:, 2]) / 2, np.minimum(prob[:, 1], prob[:, 2])


def lane_change_prob(raw):
  # desire_state comes out of the model as probabilities already
  return raw[:, 2566:2574][:, 3:5].max(1)


def target_lat(plan, center, alpha):
  """Lateral plan pulled toward lane center: returns (33, 3) of y, yaw, yaw rate."""
  x, y = plan[:, 0], plan[:, 1]
  xc = np.minimum(x, MAX_X)
  off = y - np.interp(xc, X_IDXS, center)
  ramp = np.clip(T / RAMP_T, 0, 1)
  y2 = y - alpha * ramp * off
  yaw2 = np.arctan2(np.gradient(y2), np.maximum(np.gradient(x), 1e-3))
  yaw2 = plan[:, 11] + (yaw2 - np.arctan2(np.gradient(y), np.maximum(np.gradient(x), 1e-3)))  # keep the model's own yaw, add the change
  yr2 = plan[:, 14] + (np.gradient(yaw2, T) - np.gradient(plan[:, 11], T))
  return np.stack([y2, yaw2, yr2], axis=1)


def load(alpha):
  P = np.load(f'{HERE}/head_weights.npz')
  scale = P[H + 'scale_layer.plan.scale'][ROWS]
  data = []
  for p in sorted(glob.glob(f'{HERE}/seg/*.npz')):
    name = os.path.basename(p)
    if name.split('_')[0] not in ('fe', 'fd', 'f7', 'id4', 'n76', 'n77', 'b2d', 'be', 's39', 's606', 's608'):  # base replays only
      continue
    d = np.load(p)
    center, prob = lane_center_all(d['raw'])
    ok = (prob > 0.5) & (d['v_ego'] > 8) & (lane_change_prob(d['raw']) < 0.1)
    plan = d['plan']
    cur = np.concatenate([plan[:, :, c] for c in LAT_CH], axis=1)  # (N, 99) ordered as ROWS
    tgt = cur.copy()
    for i in np.flatnonzero(ok):
      tl = target_lat(plan[i], center[i], alpha)
      tgt[i] = np.concatenate([tl[:, 0], tl[:, 1], tl[:, 2]])
    keep = d['v_ego'] > 3
    dy = np.where(scale > 0, (tgt - cur) / np.where(scale > 0, scale, 1), 0.0)  # y at t=0 has scale 0: fixed at 0 by design
    data.append(dict(path=p, route=str(d['seg'][0]).rsplit('/', 1)[0], h=d['x_plan_trunk'][keep], dy=dy[keep], keep=keep, ok=ok[keep]))
  return data, scale


def fit(parts, lam, w_lo):
  Hm = np.concatenate([np.concatenate([p['h'], np.ones((len(p['h']), 1))], 1) for p in parts])
  Y = np.concatenate([p['dy'] for p in parts])
  # frames without confident lanes keep their plan: weight them up so the edit stays off there
  sw = np.sqrt(np.concatenate([np.where(p['ok'], 1.0, w_lo) for p in parts]))[:, None]
  Hm, Y = Hm * sw, Y * sw
  A = Hm.T @ Hm + lam * len(Hm) * np.eye(Hm.shape[1])
  A[-1, -1] -= lam * len(Hm)  # do not shrink the bias
  return np.linalg.solve(A, Hm.T @ Y)  # (257, 99)


def steer_curv(plans, v_ego, lat_t=0.25):
  """Desired curvature the controller would get from each plan (as modeld does it, before smoothing)."""
  return np.array([get_curvature_from_plan(p[:, 11], p[:, 14], T, v, lat_t) for p, v in zip(plans, v_ego)])


def apply(d, D, scale):
  h = np.concatenate([d['x_plan_trunk'], np.ones((len(d['x_plan_trunk']), 1))], 1)
  dmu = (h @ D) * scale
  plan = d['plan'].copy()
  for k, c in enumerate(LAT_CH):
    plan[:, :, c] += dmu[:, k * 33:(k + 1) * 33]
  return plan


def main():
  tag, alpha = sys.argv[1], float(sys.argv[2])
  lam = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
  w_lo = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
  data, scale = load(alpha)
  routes = sorted({p['route'] for p in data})
  print(f'alpha {alpha} lam {lam} w_lo {w_lo}: {len(data)} segments, {len(routes)} routes')
  verbose = os.environ.get('VERBOSE')
  pool = {'in0': [], 'in1': [], 'dk_lo': [], 'dk_hi': [], 'dy_lo': []}
  for r in routes:
    D = fit([p for p in data if p['route'] != r], lam, w_lo)
    for p in [p for p in data if p['route'] == r]:
      base = np.load(p['path'])
      d = dict(base)
      d['plan'] = apply(d, D, scale).astype(np.float32)
      ins0, ins1 = inside_offset(base), inside_offset(d)
      la = np.abs(d['curv'] * d['v_ego'] ** 2)
      m = (ins0[2] > 0.5) & (d['v_ego'] > 8) & (la > 0.7)
      lo = (ins0[2] < 0.5) & (d['v_ego'] > 3)
      dk = np.abs(steer_curv(d['plan'], d['v_ego']) - steer_curv(base['plan'], d['v_ego']))
      pool['in0'].append(ins0[0][m, 1]); pool['in1'].append(ins1[0][m, 1])
      pool['dk_lo'].append(dk[lo]); pool['dk_hi'].append(dk[(ins0[2] > 0.5) & (d['v_ego'] > 3)])
      pool['dy_lo'].append(np.abs(d['plan'][lo][:, 16, 1] - base['plan'][lo][:, 16, 1]))
      pool.setdefault('dk_lo_fast', []).append(dk[lo & (d['v_ego'] > 8)])
      pool.setdefault('dk_slow', []).append(dk[d['v_ego'] < 8])
      seg = os.path.basename(p['path'])[:-4]
      if verbose and m.sum() > 10:
        print(f'  held out {seg:>8}: curves n={m.sum():4d} inside@20m median {np.median(ins0[0][m, 1]):+.2f} -> {np.median(ins1[0][m, 1]):+.2f}, '
              f'p90 {np.percentile(ins0[0][m, 1], 90):+.2f} -> {np.percentile(ins1[0][m, 1], 90):+.2f}')
      np.savez_compressed(f'{HERE}/seg/cut{tag}_{seg}.npz', **d)
  P = {k: np.concatenate(v) for k, v in pool.items()}
  print(f'  HELD OUT, confident-lane curves (n={len(P["in0"])}): inside@20m median {np.median(P["in0"]):+.2f} -> {np.median(P["in1"]):+.2f}, '
        f'p90 {np.percentile(P["in0"], 90):+.2f} -> {np.percentile(P["in1"], 90):+.2f}, mean |inside| {np.abs(P["in0"]).mean():.2f} -> {np.abs(P["in1"]).mean():.2f}')
  print(f'  steering change |d curvature| p95: confident lanes {np.percentile(P["dk_hi"], 95):.4f}, NOT confident {np.percentile(P["dk_lo"], 95):.4f} '
        f'(path change @2.5 s p95 {np.percentile(P["dy_lo"], 95):.2f} m)')
  q = lambda a: f'p95 {np.percentile(a, 95):.4f} p99.9 {np.percentile(a, 99.9):.4f} max {a.max():.4f}'
  print(f'  AT SPEED (v > 8) not-confident steering change: {q(P["dk_lo_fast"])} | v < 8: {q(P["dk_slow"])}')
  np.savez(f'{HERE}/cut{tag}_delta.npz', D=fit(data, lam, w_lo), rows=ROWS, scale=scale, alpha=alpha, lam=lam)


if __name__ == '__main__':
  main()
