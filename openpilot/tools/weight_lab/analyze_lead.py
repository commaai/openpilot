#!/usr/bin/env python3
"""Lead speed shortcut: does the lead head under-read lead speed that its own features carry?

Ground truth is radar leadOne (radar-matched) from radarState. Fit segments and test segments are split
by segment so a probe cannot memorize one scene.

usage: analyze_lead.py A.npz,B.npz,... FIT_SEGS   (FIT_SEGS: comma list of segment numbers, the rest is test)
"""
import sys
import numpy as np

from openpilot.tools.weight_lab.lab_paths import DATA

LEAD = slice(917, 1061)
LEAD_PROB = slice(1061, 1064)


def lead_pred(raw):
  mu = raw[:, LEAD][:, :72].reshape(-1, 3, 6, 4)
  prob = 1 / (1 + np.exp(-raw[:, LEAD_PROB][:, 0]))
  return mu[:, 0, 0, 0], mu[:, 0, 0, 2], prob  # x, v at t=0 for the first lead, prob


def ridge(X, y, lam=1.0):
  Xm, ym = X.mean(0), y.mean()
  Xc = X - Xm
  w = np.linalg.solve(Xc.T @ Xc + lam * len(X) * np.eye(X.shape[1]), Xc.T @ (y - ym))
  return w, ym - Xm @ w


def slope_report(name, pred, v_lead, v_ego):
  dv_true = v_lead - v_ego
  dv_pred = pred - v_ego
  k = np.polyfit(dv_true, dv_pred, 1)[0]
  err = np.abs(pred - v_lead)
  rows = []
  for lo, hi in [(-30, -5), (-5, -2), (-2, 2), (2, 5), (5, 30)]:
    m = (dv_true >= lo) & (dv_true < hi)
    if m.sum() > 20:
      rows.append(f'[{lo},{hi}) n={m.sum()} MAE {err[m].mean():.2f} bias {np.mean(pred[m] - v_lead[m]):+.2f}')
  print(f'{name:>18}: slope of (pred-vEgo) on (vLead-vEgo) = {k:.2f}  (1 = reads lead, 0 = copies vEgo) | MAE {err.mean():.2f}')
  for r in rows:
    print(' ' * 22 + r)


def main():
  ds = [np.load(p) for p in sys.argv[1].split(',')]
  d = {k: np.concatenate([x[k] for x in ds]) for k in ds[0].files}
  fit_segs = {int(s) for s in sys.argv[2].split(',')}
  seg_num = np.array([int(s.rsplit('/', 1)[1]) for s in d['seg']])
  radar = d['radar']
  ok = (radar[:, 0] > 0.5) & (radar[:, 1] < 80)
  x_pred, v_pred, prob = lead_pred(d['raw'])
  v_ego = d['v_ego']
  v_lead, d_rel = radar[:, 2], radar[:, 1]
  ok &= (prob > 0.5) & (np.abs(x_pred - d_rel) < 0.25 * d_rel + 3)  # model sees the same car as radar
  fit = ok & np.isin(seg_num, list(fit_segs))
  test = ok & ~np.isin(seg_num, list(fit_segs))
  print(f'radar-matched frames: fit {fit.sum()}, test {test.sum()}; test |vLead-vEgo|>3 frames: {(test & (np.abs(v_lead - v_ego) > 3)).sum()}')

  slope_report('model head (test)', v_pred[test], v_lead[test], v_ego[test])
  for tap in ('lead_trunk', 'lead_feat'):
    X = d[f'x_{tap}']
    w, b = ridge(X[fit], v_lead[fit], lam=1e-3)
    slope_report(f'probe {tap}', X[test] @ w + b, v_lead[test], v_ego[test])

  # where does the head point? decompose its readout direction in the 64-d trunk
  X = d['x_lead_trunk']
  w_ego, _ = ridge(X[fit], v_ego[fit], lam=1e-3)
  w_dv, _ = ridge(X[fit], (v_lead - v_ego)[fit], lam=1e-3)
  P = np.load(f'{DATA}/head_weights.npz')
  Hh = 'vision_model.policy.hydra.'
  Wf = P[Hh + 'final_layer.lead.weight'] * P[Hh + 'scale_layer.lead.scale'][:, None]
  w_head = Wf[2]  # lead 0, t=0, v
  cos = lambda a, c: float(a @ c / np.linalg.norm(a) / np.linalg.norm(c))
  print(f'\nin the 64-d lead trunk: cos(head v readout, vEgo direction) {cos(w_head, w_ego):+.2f}, '
        f'cos(head v readout, (vLead-vEgo) direction) {cos(w_head, w_dv):+.2f}, cos(vEgo dir, dv dir) {cos(w_ego, w_dv):+.2f}')
  # how much of the head's prediction variance on test frames comes from each part
  h = X[test]
  pe = (h @ w_head)
  print(f'head v output: corr with vEgo {np.corrcoef(pe, v_ego[test])[0, 1]:+.2f}, corr with radar vLead {np.corrcoef(pe, v_lead[test])[0, 1]:+.2f}, '
        f'corr of (pred - vEgo) with (vLead - vEgo) {np.corrcoef(v_pred[test] - v_ego[test], (v_lead - v_ego)[test])[0, 1]:+.2f}')


if __name__ == '__main__':
  main()
