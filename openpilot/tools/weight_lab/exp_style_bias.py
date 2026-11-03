#!/usr/bin/env python3
"""Add a bias along the style probe directions (exp_style_probe.py) and look at what the plan does.

b_in += W_in @ (alpha * sigma * dir * x_std). alpha is in units of the natural spread (sigma) of x along dir on the
111 base segments. The change is exact in the plan head (head.forward), on master weights.
Also checks if the direction is a slow trait (persists over tens of seconds) or a fast scene signal.
"""
import numpy as np

from openpilot.tools.weight_lab import head
from openpilot.tools.weight_lab.exp_style_probe import OUT, COL_X, COL_VX, COL_AX, at, load_all, situations

ALPHAS = (-2.0, -1.0, 1.0, 2.0)
LAGS_S = (1, 5, 15, 30, 60)
HZ = 20
N_RANDOM = 20


def stop_mask(D):
  """Approach to a stop: moving, and the plan is near 0 m/s at 10 s."""
  vx10 = D['plan'][:, -1, COL_VX]
  return (D['v'] > 5) & (vx10 < 1.0)


def summary(plan, v):
  vx, px, ax = plan[:, :, COL_VX], plan[:, :, COL_X], plan[:, :, COL_AX]
  return dict(a0=ax[:, 0], dv2=at(vx, 2.0) - vx[:, 0], vx6_v=at(vx, 6.0) - v, x4=at(px, 4.0), x10=px[:, -1],
              vx1=at(vx, 1.0))


def biased(P, delta_x):
  Q = {k: v.copy() for k, v in P.items()}
  Q['in_layer.plan.bias'] = Q['in_layer.plan.bias'] + Q['in_layer.plan.weight'] @ delta_x
  return Q


def autocorr(p, seg_id, lag):
  same = seg_id[lag:] == seg_id[:-lag]
  a, b = p[lag:][same], p[:-lag][same]
  return np.corrcoef(a, b)[0, 1]


def main():
  D = load_all()
  S = situations(D)
  S['stop'] = stop_mask(D)
  Z = np.load(f'{OUT}/probes.npz')
  x_mean, x_std = Z['x_mean'], Z['x_std']
  dirs = {k.replace('dir_', ''): Z[k] for k in Z.files if k.startswith('dir_')}
  combo = sum(dirs.values())
  dirs['combo'] = combo / np.linalg.norm(combo)
  rng = np.random.default_rng(0)
  zs = (D['x'] - x_mean) / x_std

  # slow trait or fast scene signal? autocorrelation of the projection inside a segment
  seg_id =np.repeat(np.arange(len(head.base_files())), [len(np.load(p, mmap_mode='r')['v_ego']) for p in head.base_files()])
  print('== autocorrelation over lag (same segment). a driver trait would stay high at 30-60 s')
  ref = dict(v_ego=D['v'], desired_accel=D['accel'])
  for name in ('combo', 'launch', 'follow', 'cruise'):
    ref[f'proj_{name}'] = zs @ dirs[name]
  for name, p in ref.items():
    print(f'  {name:14s} ' + ' '.join(f'{lag}s={autocorr(p, seg_id, lag * HZ):+.2f}' for lag in LAGS_S))
  p = ref['proj_combo']
  print(f'  combo projection: corr with v_ego {np.corrcoef(p, D["v"])[0, 1]:+.2f}, with desired accel'
        f' {np.corrcoef(p, D["accel"])[0, 1]:+.2f}, with plan vx(6s)-v {np.corrcoef(p, at(D["plan"][:, :, COL_VX], 6.0) - D["v"])[0, 1]:+.2f}')
  route_means = {r: p[D['route'] == r].mean() for r in np.unique(D['route'])}
  print('  combo projection mean by route (sigma units): ' + ' '.join(f'{r}={m / p.std():+.2f}' for r, m in route_means.items()))

  P = head.master_weights()
  idx_all = {k: np.where(m)[0][::4] for k, m in S.items()}
  base = {k: summary(head.forward(P, D['x'][i])['plan'], D['v'][i]) for k, i in idx_all.items()}
  print('\n== bias along a direction: mean plan change per situation (frames, every 4th)')
  print('  launch dv2 = plan dv over 2 s | cruise vx6-v | follow x4 = plan distance at 4 s (more = smaller gap)'
        ' | stop vx1 (more = later braking), x10 (stop point moves)')
  for name in ('combo', 'launch', 'follow', 'cruise'):
    sigma = p.std() if name == 'combo' else (zs @ dirs[name]).std()
    for alpha in ALPHAS:
      Q = biased(P, alpha * sigma * dirs[name] * x_std)
      out = {k: summary(head.forward(Q, D['x'][i])['plan'], D['v'][i]) for k, i in idx_all.items()}

      def dm(sit, key):
        return np.mean(out[sit][key] - base[sit][key])

      print(f'  {name:6s} a={alpha:+.0f}: launch dv2 {dm("launch", "dv2"):+.2f} a0 {dm("launch", "a0"):+.2f} | cruise vx6-v'
            f' {dm("cruise", "vx6_v"):+.2f} a0 {dm("cruise", "a0"):+.2f} | follow x4 {dm("follow", "x4"):+.2f} a0 {dm("follow", "a0"):+.2f}'
            f' | stop vx1 {dm("stop", "vx1"):+.2f} x10 {dm("stop", "x10"):+.1f} a0 {dm("stop", "a0"):+.2f}')
  # random directions with the same size, for a noise floor
  sigma = p.std()
  rand = []
  for _ in range(N_RANDOM):
    r = rng.standard_normal(512)
    r /= np.linalg.norm(r)
    Q = biased(P, 2.0 * sigma * r * x_std)
    out = {k: summary(head.forward(Q, D['x'][i])['plan'], D['v'][i]) for k, i in idx_all.items()}
    rand.append([np.mean(out[s][k] - base[s][k]) for s, k in (('launch', 'dv2'), ('cruise', 'vx6_v'), ('follow', 'x4'),
                                                                ('stop', 'vx1'), ('stop', 'x10'))])
  rand = np.abs(np.array(rand))
  print(f'  random dirs a=+2 (|mean change|, median of {N_RANDOM}): launch dv2 {np.median(rand[:, 0]):.2f} cruise vx6-v'
        f' {np.median(rand[:, 1]):.2f} follow x4 {np.median(rand[:, 2]):.2f} stop vx1 {np.median(rand[:, 3]):.2f} x10 {np.median(rand[:, 4]):.1f}')
  print('  frames: ' + ' '.join(f'{k}={len(i)}' for k, i in idx_all.items()))

  # does the change follow the plan's own shape (a gain, like post-processing) or something new?
  name, alpha = 'combo', 2.0
  Q = biased(P, alpha * sigma * dirs[name] * x_std)
  print(f'\n== combo a=+2: per frame, is the accel change a gain on the plan accel? (fit da0 = g * a0 + c)')
  for sit, i in idx_all.items():
    a_base = base[sit]['a0']
    da = summary(head.forward(Q, D['x'][i])['plan'], D['v'][i])['a0'] - a_base
    A = np.column_stack([a_base, np.ones_like(a_base)])
    coef, *_ = np.linalg.lstsq(A, da, rcond=None)
    r2 = 1 - np.var(da - A @ coef) / max(np.var(da), 1e-12)
    print(f'  {sit:6s}: da0 mean {da.mean():+.3f} std {da.std():.3f} p5/p95 {np.percentile(da, 5):+.2f}/{np.percentile(da, 95):+.2f}'
          f' | gain fit g {coef[0]:+.2f} c {coef[1]:+.2f} R2 {r2:.2f}')


if __name__ == '__main__':
  main()
