#!/usr/bin/env python3
"""Narrow streets: does the plan go too fast on residential streets with parked cars, and can a gated circuit fix it?

Steps (all on the 111 base replay segments, 20 Hz, master model):
  1. scene features per frame from the model outputs (lane lines, road edges) and engagement from the qlogs
  2. camera-checked labels: RES = residential streets with cars parked along the side (thumbnail sheets)
  3. human vs plan: future real speed vs plan vx at 3 s / 6 s, split by engaged and human driven
  4. probes on x_plan_feat, held out by route, with speed controls (Frisch-Waugh) and speed-matched weights
  5. circuit in the fork units (trunk 48): plan vx pulled down by (v - V_MIN) when the residential gate is on
     scored held out by route against the shipped stack: |d accel| > 0.1 / 0.3, p99.9, max
  6. code rule: the same cap from the modelV2 lane / road edge outputs, with time smoothing

usage: exp_narrow.py [--save]   (--save writes patches/narrow_v1.npz from all routes)
"""
import os
import sys
from multiprocessing import Pool
import numpy as np

from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.weight_lab import head
from openpilot.tools.weight_lab.lab_paths import CODE, DATA
from openpilot.tools.weight_lab.lights_edit import accels
from openpilot.tools.weight_lab.slots import Circuit, N_PLAN

OUT = f'{DATA}/exp/narrow'
X_IDXS = np.array([192.0 * (i / 32) ** 2 for i in range(33)])
T = np.array([10 * (i / 32) ** 2 for i in range(33)])
HZ = 20
COL_X, COL_VX, COL_AX = 0, 3, 6
LANE, LANE_PROB, EDGE = slice(117, 645), slice(645, 653), slice(653, 917)

# camera-checked residential streets with cars parked along the side: (route, segment, frame from, frame to)
RES = [('f7', 9, 700, 900), ('f7', 13, 800, 1100), ('f7', 14, 0, 200), ('f7', 19, 300, 1199), ('f7', 20, 0, 1199),
       ('f7', 21, 0, 1199), ('fd', 0, 400, 1199), ('fd', 4, 640, 1199), ('fd', 5, 0, 300), ('fe', 7, 600, 1199),
       ('fe', 8, 0, 1199), ('s606', 11, 500, 1199), ('s606', 12, 0, 1199), ('s608', 1, 0, 1199), ('s608', 2, 0, 300),
       ('s608', 12, 640, 1199), ('s608', 13, 0, 1199), ('s608', 26, 500, 1199), ('s608', 27, 0, 1199)]

V_MIN = 9.0       # m/s, no cap below this speed
GAIN = 0.5        # plan vx at TAU and later is pulled down by GAIN * (v - V_MIN)
TAU = 2.0         # s, the pull ramps in over this time
CAP = 3.0         # m/s, max (v - V_MIN) the circuit reads
V_OFF = 12.0      # m/s, the circuit fades out from here to V_OFF + 1 (residential p99 speed in the data is 11.6)
FP_LO, FP_HI = 0.01, 0.002  # gate is 0 / 1 at these false alarm rates on non-residential frames > V_MIN
LAM = 0.1


def at(rows, t, xs=T):
  return np.array([np.interp(t, xs, r) for r in rows])


def shift(a, k):
  out = np.full(len(a), np.nan)
  if k < len(a):
    out[:len(a) - k] = a[k:]
  return out


def scan_qlog(seg):
  out = {'t_en': [], 'en': [], 't_cs': [], 'gas': [], 'vcruise': []}
  for m in LogReader(f'{seg}/q', sort_by_time=True):
    if m.which() == 'selfdriveState':
      out['t_en'].append(m.logMonoTime)
      out['en'].append(m.selfdriveState.enabled)
    elif m.which() == 'carState':
      out['t_cs'].append(m.logMonoTime)
      out['gas'].append(m.carState.gasPressed)
      out['vcruise'].append(m.carState.vCruise)
  return seg, {k: np.array(v) for k, v in out.items()}


def engagement(segs):
  path = f'{OUT}/engaged.npy'
  if not os.path.exists(path):
    with Pool(8) as p:
      np.save(path, dict(p.map(scan_qlog, segs)), allow_pickle=True)
  return np.load(path, allow_pickle=True).item()


def scene(d):
  """Per frame features from the model outputs that the planner can also read from modelV2."""
  raw = d['raw']
  lanes = raw[:, LANE][:, :264].reshape(-1, 4, 33, 2)[..., 0]
  prob = 1 / (1 + np.exp(-raw[:, LANE_PROB].reshape(-1, 4, 2)[..., 1]))
  edges = raw[:, EDGE][:, :132].reshape(-1, 2, 33, 2)[..., 0]
  edge_std = np.exp(raw[:, EDGE][:, 132:].reshape(-1, 2, 33, 2)[..., 0])
  f = {}
  for dist in (10, 20, 30):
    el, er = at(edges[:, 0], dist, X_IDXS), at(edges[:, 1], dist, X_IDXS)
    ll, lr = at(lanes[:, 1], dist, X_IDXS), at(lanes[:, 2], dist, X_IDXS)
    f[f'el{dist}'], f[f'er{dist}'], f[f'ew{dist}'] = el, er, np.abs(er - el)
    f[f'lw{dist}'] = np.abs(lr - ll)
    f[f'es{dist}'] = at(edge_std[:, 0], dist, X_IDXS) + at(edge_std[:, 1], dist, X_IDXS)
  f['lp'] = prob[:, 1:3].min(1)
  f['lp_outer'] = prob[:, [0, 3]].max(1)
  return f


def load():
  path = f'{OUT}/exp_narrow_cache.npz'
  if os.path.exists(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}
  files = head.base_files()
  E = engagement(sorted({str(np.load(p)['seg'][0]) for p in files}))
  rows = {}
  for p in files:
    d = np.load(p)
    v = d['v_ego'].astype(np.float64)
    e = E[str(d['seg'][0])]
    i = np.clip(np.searchsorted(e['t_en'], d['t']) - 1, 0, len(e['t_en']) - 1)
    j = np.clip(np.searchsorted(e['t_cs'], d['t']) - 1, 0, len(e['t_cs']) - 1)
    segn = int(str(d['seg'][0]).rsplit('/', 1)[1])
    route = os.path.basename(p).split('_')[0]
    res = np.zeros(len(v), bool)
    for r, s, a, b in RES:
      if r == route and s == segn:
        res |= (d['frame'] >= a) & (d['frame'] <= b)
    cols = scene(d) | {
      'v': v, 'accel': d['accel'], 'curv': d['curv'], 'x': d['x_plan_feat'], 'frame': d['frame'],
      'pv3': at(d['plan'][:, :, COL_VX], 3.0), 'pv6': at(d['plan'][:, :, COL_VX], 6.0),
      'rv3': shift(v, 3 * HZ), 'rv6': shift(v, 6 * HZ),
      'lead': (d['radar'][:, 0] > 0.5) & (d['radar'][:, 1] < 60),
      'en': e['en'][i], 'gas': e['gas'][j], 'vcruise': e['vcruise'][j] / 3.6,
      'res': res, 'route': np.full(len(v), route), 'seg_i': np.full(len(v), files.index(p)),
    }
    for k, val in cols.items():
      rows.setdefault(k, []).append(val)
  D = {k: np.concatenate(v) for k, v in rows.items()}
  np.savez(path, **D)
  return D


def ridge(A, y, lam, w=None):
  """Weighted ridge; the last column of A is the intercept and is not penalized."""
  w = np.ones(len(y)) if w is None else w
  pen = lam * len(A) * np.eye(A.shape[1])
  pen[-1, -1] = 0.0
  return np.linalg.solve(A.T @ (A * w[:, None]) + pen, (A * w[:, None]).T @ y)


def auc(s, y):
  o = np.argsort(s)
  r = np.empty(len(s))
  r[o] = np.arange(len(s))
  p = y.sum()
  n = len(y) - p
  return (r[y].sum() - p * (p - 1) / 2) / (p * n) if p and n else np.nan


def speed_matched(y, v, rt, m):
  """Weights so that residential and other frames have the same speed mix (1 m/s bins) in each route."""
  w = np.zeros(len(y))
  b = np.floor(v).astype(int)
  for r in np.unique(rt[m]):
    for k in np.unique(b[m & (rt == r)]):
      s = m & (rt == r) & (b == k)
      pos, neg = s & y, s & ~y
      if pos.any() and neg.any():
        w[pos], w[neg] = 0.5 / pos.sum(), 0.5 / neg.sum()
      else:
        w[s] = 0.1 / s.sum()
  return w / w[m].mean()


def human_gap(D):
  print('\n== 3. human vs plan (frames with no radar lead < 60 m). rv = real speed later, pv = plan vx')
  v, R = D['v'], D['res']
  human = ~D['en'] | D['gas']
  wide = ~R & (D['ew20'] > 12) & (D['lp'] > 0.5)
  for lo, hi in ((6, 9), (9, 12), (12, 15)):
    base = (v >= lo) & (v < hi) & ~D['lead']
    for name, m in (('residential engaged', R & ~human), ('residential human', R & human), ('wide engaged', wide & ~human),
                    ('wide human', wide & human), ('other engaged', ~R & ~wide & ~human), ('other human', ~R & ~wide & human)):
      m = m & base
      if m.sum() < 40:
        continue
      print(f'  v {lo:2d}-{hi:2d} {name:20s} n {m.sum():5d} ({m.sum() / HZ:4.0f} s) | pv3-v {np.median(D["pv3"][m] - v[m]):+.2f} | '
            f'rv3-pv3 {np.nanmedian(D["rv3"][m] - D["pv3"][m]):+.2f} | rv6-pv6 {np.nanmedian(D["rv6"][m] - D["pv6"][m]):+.2f} | '
            f'set speed - pv6 {np.median(D["vcruise"][m] - D["pv6"][m]):+.1f}')
  m = R & (v > 2)
  print(f'  residential speed (v > 2): p50 {np.median(v[m]):.1f} p90 {np.percentile(v[m], 90):.1f} p99 {np.percentile(v[m], 99):.1f} m/s, '
        f'plan vx@3s p90 {np.percentile(D["pv3"][m], 90):.1f}, set speed p50 {np.median(D["vcruise"][m]):.1f} | '
        f'share v > {V_MIN:.0f}: {np.mean(v[m] > V_MIN):.2f}, > 11: {np.mean(v[m] > 11):.3f}')


def probes(D):
  print('\n== 4. probes on x_plan_feat, leave one route out')
  idx = np.flatnonzero((D['v'] > 4) & (D['frame'] % 2 == 0))
  x = D['x'][idx].astype(np.float64)
  X = np.column_stack([(x - x.mean(0)) / (x.std(0) + 1e-6), np.ones(len(idx))])
  v, rt, y = D['v'][idx], D['route'][idx], D['res'][idx]
  ctrl = np.column_stack([v, v ** 2, v ** 3, D['accel'][idx], D['curv'][idx] * v ** 2, np.abs(D['curv'][idx]) * v ** 2,
                          D['lead'][idx], np.ones(len(idx))])
  Xr = X[:, :-1] - ctrl @ np.linalg.lstsq(ctrl, X[:, :-1], rcond=None)[0]
  Xr = np.column_stack([Xr, np.ones(len(idx))])
  for name in ('ew20', 'er20', 'lw20'):
    t = np.clip(D[name][idx], 0, 25)
    tr = t - ctrl @ np.linalg.lstsq(ctrl, t, rcond=None)[0]
    pred = np.zeros(len(idx))
    for r in np.unique(rt):
      te = rt == r
      pred[te] = Xr[te] @ ridge(Xr[~te], tr[~te], 1e-3)
    print(f'  {name}: speed/accel/curv controls explain R2 {1 - tr.var() / t.var():.2f}; x (speed removed) predicts the rest with LORO R2 '
          f'{1 - np.mean((tr - pred) ** 2) / tr.var():.2f}')
  pred = np.zeros(len(idx))
  for r in np.unique(rt):
    te = rt == r
    pred[te] = X[te] @ ridge(X[~te], y[~te].astype(float), LAM, speed_matched(y, v, rt, ~te)[~te])
  fast = v > V_MIN
  print(f'  residential label ({y.sum()} frames, routes {", ".join(np.unique(rt[y]))}): AUC of low speed alone {auc(-v, y):.3f}')
  print(f'  speed-matched probe: AUC v>{V_MIN:.0f} {auc(pred[fast], y[fast]):.3f}, v 6-9 {auc(pred[(v > 6) & (v < 9)], y[(v > 6) & (v < 9)]):.3f}, '
        f'v 9-12 {auc(pred[fast & (v < 12)], y[fast & (v < 12)]):.3f} | corr(probe, v) {np.corrcoef(pred, v)[0, 1]:+.2f} | per held-out route AUC '
        + ' '.join(f'{r} {auc(pred[rt == r], y[rt == r]):.2f}' for r in np.unique(rt[y])))
  return idx, pred


def fit_gate(D, train):
  """Speed and residential readers on raw x, fit on the train frames. Returns raw-x weights (w, b) for both."""
  idx = np.flatnonzero(train & (D['frame'] % 2 == 0))
  x = D['x'][idx].astype(np.float64)
  mu, sd = x.mean(0), x.std(0) + 1e-6
  X = np.column_stack([(x - mu) / sd, np.ones(len(idx))])
  v, rt, y = D['v'][idx], D['route'][idx], D['res'][idx]

  def raw(beta):
    return beta[:-1] / sd, beta[-1] - np.sum(beta[:-1] * mu / sd)
  wv, bv = raw(ridge(X, v, 1e-3))  # all speeds, so standstill does not read as fast
  moving = v > 4
  wp, bp = raw(ridge(X[moving], y[moving].astype(float), LAM, speed_matched(y[moving], v[moving], rt[moving], np.ones(moving.sum(), bool))))
  neg = (v > V_MIN) & ~y
  p = x @ wp + bp
  lo, hi = np.quantile(p[neg], 1 - FP_LO), np.quantile(p[neg], 1 - FP_HI)
  return wv, bv, wp / (hi - lo), (bp - lo) / (hi - lo)


def build(wv, bv, wg, bg, scale):
  c = Circuit('fork')
  s1, s2, s3, g1, g2 = c.free('r4')[:5]
  c.r4(s1, wv, bv - V_MIN)        # relu(v - V_MIN)
  c.r4(s2, wv, bv - V_MIN - CAP)  # relu(v - V_MIN - CAP), so s1 - s2 is the speed excess clamped to [0, CAP]
  c.r4(s3, wv, bv - V_OFF)        # relu(v - V_OFF), turns the circuit off at high speed
  c.r4(g1, wg, bg)                # gate, 0 at FP_LO and 1 at FP_HI
  c.r4(g2, wg, bg - 1.0)          # gate - 1, so g1 - g2 is the gate clamped to [0, 1]
  k = c.free('r6')[0]
  # AND: relu(speed excess - CAP * (1 - gate) - CAP * (v - V_OFF)), 0 when the gate is off or v > V_OFF + 1
  c.r6(k, {s1: 1.0, s2: -1.0, s3: -CAP, g1: CAP, g2: -CAP}, -CAP)
  dv = -GAIN * np.minimum(T / TAU, 1.0)          # plan change per m/s of gated speed excess
  da = np.where(T < TAU, -GAIN / TAU, 0.0)
  dx = np.concatenate([[0.0], np.cumsum(0.5 * (dv[1:] + dv[:-1]) * np.diff(T))])
  readout = np.zeros(N_PLAN)
  for col, delta in ((COL_X, dx), (COL_VX, dv), (COL_AX, da)):
    rows = np.arange(33) * 15 + col
    ok = np.abs(scale[rows]) > 1e-6
    readout[rows[ok]] = delta[ok] / scale[rows[ok]]
  c.out(48, {k: 1.0}, readout)
  return c


def tails(da, name):
  if len(da) == 0:
    return f'{name}: none'
  return (f'{name}: n {len(da)} | |da|>0.1 {100 * np.mean(np.abs(da) > 0.1):.2f}% >0.3 {100 * np.mean(np.abs(da) > 0.3):.2f}% | '
          f'p99.9 {np.percentile(np.abs(da), 99.9):.3f} max {np.abs(da).max():.3f} | mean {da.mean():+.4f}')


def score(D, files, patch_for_route):
  """Accel change vs the shipped stack on each segment, with the patch fit without that segment's route."""
  base_P = head.patched_weights(head.SHIPPED)
  new_P = {r: head.patched_weights(head.SHIPPED + [p]) for r, p in patch_for_route.items()}
  da = np.zeros(len(D['v']))
  for si, p in enumerate(files):
    m = D['seg_i'] == si
    d = np.load(p)
    route = os.path.basename(p).split('_')[0]
    a0 = accels(head.edited_plan(d, base_P))
    a1 = accels(head.edited_plan(d, new_P[route]))
    da[m] = a1 - a0
  return da


def report(D, da, title):
  v, R = D['v'], D['res']
  wide = ~R & (D['ew20'] > 12) & (D['lp'] > 0.5) & (v > V_MIN)
  print(f'\n== {title}')
  print('  ' + tails(da[wide], 'open wide roads v>9'))
  print('  ' + tails(da[~R & (v > V_MIN)], 'all non-residential v>9'))
  print('  ' + tails(da[v > 20], 'all frames v>20 (highway)'))
  print('  ' + tails(da[R & (v > V_MIN)], 'residential v>9 (target)'))
  print('  ' + tails(da[R], 'residential all speeds'))
  m = R & (v > V_MIN)
  print(f'  target: frames with braking > 0.1 m/s² {100 * np.mean(da[m] < -0.1):.1f}%, median change where on {np.median(da[m][da[m] < -0.01]) if (da[m] < -0.01).any() else 0:+.3f}')
  per = [f'{r} {100 * np.mean(da[(D["route"] == r) & wide] < -0.1):.2f}%' for r in np.unique(D['route']) if ((D['route'] == r) & wide).any()]
  print('  open wide roads, braking > 0.1 per held-out route: ' + ' '.join(per))


def code_rule(D, idx_p, pred_x):
  """The same gate from the modelV2 lane / road edge outputs only, plus a 2 s moving average (code can keep state)."""
  print('\n== 6. code rule from lane line / road edge outputs (planner side), leave one route out')
  keys = ['ew10', 'ew20', 'ew30', 'el10', 'el20', 'er10', 'er20', 'lw10', 'lw20', 'es10', 'es20', 'lp', 'lp_outer']
  F = np.column_stack([np.clip(D[k], -30, 30) for k in keys])
  F = np.column_stack([(F - F.mean(0)) / F.std(0), np.ones(len(F))])
  v, rt, y = D['v'], D['route'], D['res']
  use = v > 4
  pred = np.full(len(v), np.nan)
  for r in np.unique(rt):
    te = rt == r
    tr = use & ~te
    pred[te] = F[te] @ ridge(F[tr], y[tr].astype(float), 1e-3, speed_matched(y, v, rt, tr)[tr])
  smooth = np.zeros(len(v))
  full_x = np.full(len(v), np.nan)
  full_x[idx_p] = pred_x
  for si in np.unique(D['seg_i']):
    m = np.flatnonzero(D['seg_i'] == si)
    smooth[m] = np.convolve(pred[m], np.ones(2 * HZ) / (2 * HZ), mode='full')[:len(m)]  # causal 2 s mean
  fast = v > V_MIN
  sub = np.zeros(len(v), bool)
  sub[idx_p] = True
  for name, s in (('x_plan_feat probe (weight edit)', full_x), ('lane/edge outputs, per frame', pred), ('lane/edge outputs, 2 s mean', smooth)):
    m = fast & sub & np.isfinite(s)
    neg = m & ~y
    line = f'  {name:32s} AUC v>9 {auc(s[m], y[m]):.3f}'
    for fp in (0.01, 0.002):
      thr = np.quantile(s[neg], 1 - fp)
      line += f' | TPR @ {100 * fp:.1f}% FP {np.mean(s[m & y] > thr):.2f}'
    print(line)
  return smooth


def main():
  os.makedirs(OUT, exist_ok=True)
  D = load()
  files = head.base_files()
  print(f'{len(D["v"])} frames, {len(files)} segments, routes {", ".join(np.unique(D["route"]))}; residential {D["res"].sum()} frames')
  human_gap(D)
  idx_p, pred_x = probes(D)
  scale = head.master_weights()['scale_layer.plan.scale']
  patches = {}
  for r in np.unique(D['route']):
    path = f'{OUT}/fold_{r}.npz'
    build(*fit_gate(D, D['route'] != r), scale).save(path, f'narrow fold without {r}')
    patches[r] = path
  da = score(D, files, patches)
  np.save(f'{OUT}/da_loro.npy', da)
  report(D, da, f'5. circuit, held out by route (gain {GAIN}, v_min {V_MIN}, gate FP {FP_LO}/{FP_HI})')
  code_rule(D, idx_p, pred_x)
  if '--save' in sys.argv:
    build(*fit_gate(D, np.ones(len(D['v']), bool)), scale).save(f'{CODE}/patches/narrow_v1.npz', 'narrow street speed cap (fork units)')


if __name__ == '__main__':
  main()
