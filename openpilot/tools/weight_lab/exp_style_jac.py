#!/usr/bin/env python3
"""Build a style direction by hand from plan head Jacobians, and check if it holds frame by frame.

Wanted: launch dv2 up, follow x4 up (smaller gap), stop vx1 up (later braking), stop point x10 kept, cruise vx6-v kept.
The mean Jacobian of each number over its situation frames (finite difference in standardized x) gives 5 linear
constraints. The minimum norm direction meets them on average. The real question is the spread per frame.
"""
import numpy as np

from openpilot.tools.weight_lab import head
from openpilot.tools.weight_lab.exp_style_probe import OUT, load_all, situations
from openpilot.tools.weight_lab.exp_style_bias import stop_mask, summary, biased

N_FRAMES = 300
EPS = 0.05
# (situation, number, wanted mean change)
WANT = (('launch', 'dv2', 0.5), ('follow', 'x4', 3.0), ('stop', 'vx1', 1.0), ('stop', 'x10', 0.0), ('cruise', 'vx6_v', 0.0))


def main():
  D = load_all()
  S = situations(D)
  S['stop'] = stop_mask(D)
  Z = np.load(f'{OUT}/probes.npz')
  x_std = Z['x_std']
  rng = np.random.default_rng(0)
  idx = {k: np.sort(rng.choice(np.where(m)[0], min(N_FRAMES, m.sum()), replace=False)) for k, m in S.items()}
  P = head.master_weights()
  base = {k: summary(head.forward(P, D['x'][i])['plan'], D['v'][i]) for k, i in idx.items()}

  # mean Jacobian rows, one forward per input dim with all situations stacked
  allx = np.concatenate([D['x'][idx[s]] for s in idx])
  allv = np.concatenate([D['v'][idx[s]] for s in idx])
  bounds = np.cumsum([0] + [len(idx[s]) for s in idx])
  order = list(idx)
  base_all = summary(head.forward(P, allx)['plan'], allv)
  G = np.zeros((len(WANT), 512))
  for j in range(512):
    xs = allx.copy()
    xs[:, j] += EPS * x_std[j]
    out = summary(head.forward(P, xs)['plan'], allv)
    for r, (sit, key, _) in enumerate(WANT):
      k = order.index(sit)
      sl = slice(bounds[k], bounds[k + 1])
      G[r, j] = np.mean(out[key][sl] - base_all[key][sl]) / EPS
  target = np.array([w for _, _, w in WANT])
  d = np.linalg.pinv(G) @ target
  print(f'min norm direction: |d| = {np.linalg.norm(d):.2f} std units of x ({np.linalg.norm(d) / np.sqrt(512):.3f} per dim)')

  Q = biased(P, d * x_std)
  print('per frame change (these frames were used to fit, so the means match by build):')
  for sit, key, want in WANT:
    i = idx[sit]
    delta = summary(head.forward(Q, D['x'][i])['plan'], D['v'][i])[key] - base[sit][key]
    print(f'  {sit:6s} {key:6s} want {want:+.1f}: mean {delta.mean():+.2f} std {delta.std():.2f} p5/p95 {np.percentile(delta, 5):+.2f}/{np.percentile(delta, 95):+.2f}')
  # held out frames: other frames of the same situations
  print('held out frames (every 7th frame of each situation, not the fit frames):')
  for sit, key, want in WANT:
    i = np.setdiff1d(np.where(S[sit])[0][::7], idx[sit])
    b = summary(head.forward(P, D['x'][i])['plan'], D['v'][i])[key]
    delta = summary(head.forward(Q, D['x'][i])['plan'], D['v'][i])[key] - b
    print(f'  {sit:6s} {key:6s} want {want:+.1f}: mean {delta.mean():+.2f} std {delta.std():.2f} p5/p95 {np.percentile(delta, 5):+.2f}/{np.percentile(delta, 95):+.2f} (n {len(i)})')
  i = np.arange(0, len(D['v']), 10)
  b = summary(head.forward(P, D['x'][i])['plan'], D['v'][i])
  o = summary(head.forward(Q, D['x'][i])['plan'], D['v'][i])
  print(f'all frames (every 10th, n {len(i)}): a0 change mean {np.mean(o["a0"] - b["a0"]):+.2f} p1/p99'
        f' {np.percentile(o["a0"] - b["a0"], 1):+.2f}/{np.percentile(o["a0"] - b["a0"], 99):+.2f}')
  np.save(f'{OUT}/dir_jac.npy', d)


if __name__ == '__main__':
  main()
