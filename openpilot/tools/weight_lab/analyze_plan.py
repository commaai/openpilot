#!/usr/bin/env python3
"""Read the plan head with activations from replay.py: long/lat modes and scene-selective trunk units."""
import sys
import numpy as np

from openpilot.tools.weight_lab.lab_paths import CODE, DATA as HERE  # noqa: F401
T = np.array([10 * (i / 32) ** 2 for i in range(33)])
NAMES = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'roll', 'pitch', 'yaw', 'rr', 'pr', 'yr']
H = 'policy_model.temporal_hydra.'


def load(paths):
  ds = [np.load(p) for p in paths]
  return {k: np.concatenate([d[k] for d in ds]) for k in ds[0].files}


def main():
  d = load(sys.argv[1].split(','))
  P = np.load(f'{HERE}/head_weights.npz')
  W, b, s = P[H + 'final_layer.plan.weight'], P[H + 'final_layer.plan.bias'], P[H + 'scale_layer.plan.scale']
  We = (W * s[:, None])[:495].reshape(33, 15, 256)
  h = d['x_plan_trunk']  # (N, 256), post-ReLU
  v = d['v_ego']
  accel = d['accel']
  print(f'frames {len(h)}, v_ego mean {v.mean():.1f}, accel std {accel.std():.2f}')

  # long modes: SVD of the vx readout
  U, S, Vt = np.linalg.svd(We[:, 3, :], full_matrices=False)
  z = h @ Vt[:3].T  # mode coordinates
  print('\nLONG MODES (vx readout). shape = vx change over plan time per unit of mode coordinate')
  for k in range(3):
    shape = U[:, k] * S[k]
    zc = z[:, k]
    sgn = np.sign(np.corrcoef(zc, v)[0, 1]) if k == 0 else np.sign(np.corrcoef(zc, accel)[0, 1])
    print(f' mode {k}: vx(t=0,1,2,4,6,10 s) = {(sgn * np.interp([0, 1, 2, 4, 6, 10], T, shape)).round(2)}'
          f' | corr with v_ego {np.corrcoef(zc, v)[0, 1]:+.2f}, with desired accel {np.corrcoef(zc, accel)[0, 1]:+.2f}'
          f' | coordinate std {zc.std():.2f}')

  # lat modes
  U2, S2, Vt2 = np.linalg.svd(We[:, 1, :], full_matrices=False)
  z2 = h @ Vt2[:2].T
  print('\nLAT MODES (y readout)')
  for k in range(2):
    shape = U2[:, k] * S2[k]
    print(f' mode {k}: y(t=1,2,4,6,10 s) = {np.interp([1, 2, 4, 6, 10], T, shape).round(2)}'
          f' | corr with desired curvature {np.corrcoef(z2[:, k], d["curv"])[0, 1]:+.2f}')

  # unit statistics
  active = (h > 1e-6).mean(0)
  decel = accel < -0.7
  accel_ev = accel > 0.5
  vx_w = np.linalg.norm(We[:, 3, :], axis=0)
  y_w = np.linalg.norm(We[:, 1, :], axis=0)
  print(f'\nunits: always-on(>95%) {int((active > .95).sum())}, sparse(<20%) {int((active < .2).sum())}, never-on {int((active == 0).sum())}')
  print(f'decel frames {decel.sum()} ({100 * decel.mean():.1f}%), accel frames {accel_ev.sum()}')
  # selectivity: mean activation in decel frames vs. elsewhere, for sparse-ish units
  m_dec = h[decel].mean(0)
  m_rest = h[~decel].mean(0) + 1e-6
  sel = m_dec / m_rest
  # direct effect of each unit on desired accel proxy: d vx(1 s)/d h - vx(0)
  i1 = np.searchsorted(T, 1.0)
  eff = We[i1, 3, :] - We[0, 3, :]
  order = np.argsort(-sel)
  print('\nmost decel-selective trunk units (act ratio decel/rest, on-rate, effect on vx(1s)-vx(0) per unit act, long weight, lat weight)')
  for u in order[:12]:
    print(f'  unit {u:3d}: ratio {sel[u]:6.1f}  on {100 * active[u]:5.1f}%  effect {eff[u]:+.3f}  |vx| {vx_w[u]:.2f}  |y| {y_w[u]:.2f}')
  np.savez(f'{HERE}/plan_modes.npz', Vt=Vt[:3], U=U[:, :3], S=S[:3], Vt_lat=Vt2[:2], sel=sel, active=active, eff=eff)


if __name__ == '__main__':
  main()
