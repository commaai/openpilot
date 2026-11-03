#!/usr/bin/env python3
"""Green after creeping: does the model keep braking after the light turns green, and can a gated circuit stop that?

Part 1 (measure): launches at lights from < 4 m/s (exp_green_labels.GREEN, camera-checked), the master accel and the
stop units u201 / u239 / u69 / u248 around the green.

Part 2 (circuit, owner 'stop', see slots.py): cancel the braking only when a green probe is on.
  s = w . x + b                 linear "green, go" probe on x_plan_feat (ridge; 1 = first 1 s after the green, v < 4,
                                0 = red waits and red approaches, turns on red, STOP signs, everything else)
  G = clip((s - t) / RAMP, 0, 1)               layer 1: r4 unit 47 = relu(.), r4 slot 9 = relu(. - 1)
  B = clip(-a_hat / CAP, 0, 1)                 layer 2: two r6 units on the live r4 units, a_hat = linear fit of the
                                               shipped raw accel (2 (v(0.525 s) - v0) / 0.525 - a0) on slow frames
  trunk 47 = relu((1 + BETA) G + B - 1)        ~ min(G, B), plus BETA when G = 1
  readout: + OVER * CAP m/s² of raw accel per unit (long rows: ax, vx = a t, x = a t² / 2 for t < 2 s).
With BETA = 0 and OVER = 2: when G = 1 and the model brakes by b (up to CAP), it adds 2 b, so the raw accel becomes
+b. When G = 0 it adds nothing, also when the model brakes. When the model does not brake (standing still) it adds
nothing, also when G = 1: the launch from a standstill stays the model's own.

Fit and score are leave-one-route-out: each route is scored with a circuit fit on the other 10 routes.
The GREEN / RTOR / extra labels are from the camera; the gate is fit on them, so a new drive is the real test.

usage: exp_green.py [--save]      (--save writes patches/green_go_v1.npz, only if every check passes)
env: MARGIN RAMP CAP BETA OVER (defaults = the saved patch)
"""
import argparse
import os
import numpy as np

from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan
from openpilot.tools.weight_lab import head
from openpilot.tools.weight_lab.lab_paths import CODE, DATA
from openpilot.tools.weight_lab.lights import events
from openpilot.tools.weight_lab.lights_edit import accels, T, LONG_T
from openpilot.tools.weight_lab.slots import Circuit, DEAD_R4, LIVE_R4, N_PLAN
from openpilot.tools.weight_lab.exp_stop_labels import STOP, RED
from openpilot.tools.weight_lab.exp_green_labels import GREEN, RTOR, STOP_EXTRA, RED_EXTRA

OUT = f'{DATA}/exp/green'
CACHE = f'{OUT}/cache_v1.npz'
CIRC = [69, 248, 201, 239]
V_SLOW = 4.0      # m/s: launches and probe frames are slower than this
POS_S = 20        # frames after the green thumbnail that are probe positives
LAM = 0.03        # ridge strength, probe
LAM_A = 0.01      # ridge strength, accel fit
PCT_NEG = 99.8    # threshold: this percentile of the probe on training red / STOP frames ...
PCT_OPEN = 99.95  # ... and on training open road; the larger one is used
MARGIN = float(os.environ.get('MARGIN', 0.1))  # added to the threshold
RAMP = float(os.environ.get('RAMP', 0.2))      # probe range from G = 0 to G = 1
CAP = float(os.environ.get('CAP', 0.6))        # m/s²: most braking it cancels
BETA = float(os.environ.get('BETA', 0.0))      # extra push when G = 1, in units of CAP
OVER = float(os.environ.get('OVER', 2.0))      # readout gain: > 1 gives back more than the braking (only while it brakes)
T_PUSH = 2.0      # s: the readout adds speed for this long
SLOT_B = 9
LONG_ROWS = np.array([t * 15 + c for t in range(33) for c in (0, 3, 6)])


def raw_accel(plan):
  return np.array([get_accel_from_plan(p[:, 3], p[:, 6], T, action_t=LONG_T) for p in plan])


def build_cache():
  ship = head.patched_weights(head.SHIPPED)
  master = head.master_weights()
  info = []
  for p in head.base_files():
    s = str(np.load(p)['seg'][0])
    route, segn = s.rsplit('/', 1)
    info.append((route, int(segn), p))
  keys = ('x', 'r4', 'v', 'a_m', 'a_s', 'raw_s', 'circ', 'name', 'frame', 'route', 'run', 'plan_s', 'thumb', 'stop_here')
  rows = {k: [] for k in keys}
  run, prev = -1, None
  for route, segn, p in sorted(info):
    d = np.load(p)
    if prev is None or prev != (route, segn - 1):
      run += 1  # a run is a block of back-to-back segments
    prev = (route, segn)
    n = len(d['v_ego'])
    ref = head.forward(master, d['x_plan_feat'])
    plan_s = head.edited_plan(d, ship, ref)
    stop_here = np.zeros(n, bool)
    for e in events(d):
      stop_here[d['frame'] == e['stop']] = True
    vals = (d['x_plan_feat'], ref['r4'][:, LIVE_R4].astype(np.float32), d['v_ego'], d['accel'], accels(plan_s), raw_accel(plan_s),
            d['x_plan_trunk'][:, CIRC], np.full(n, os.path.basename(p)[:-4]), d['frame'], np.full(n, route), np.full(n, run),
            plan_s.astype(np.float32), np.full(n, str(d['seg'][0]).replace('/', '_')), stop_here)
    for k, val in zip(keys, vals, strict=True):
      rows[k].append(val)
  os.makedirs(OUT, exist_ok=True)
  np.savez(CACHE, **{k: np.concatenate(v) for k, v in rows.items()})


class Data:
  def __init__(self):
    z = np.load(CACHE)
    for k in z.files:
      setattr(self, k, z[k])
    self.n = len(self.v)
    self.ac = np.zeros(self.n)  # car accel from the speed (0.5 s mean)
    for r in np.unique(self.run):
      m = np.flatnonzero(self.run == r)
      ker = np.ones(10) / 10
      vs = np.convolve(np.pad(self.v[m], (5, 4), mode='edge'), ker, mode='valid')
      self.ac[m] = np.gradient(vs) * 20

  def idx(self, name, frame):
    return int(np.flatnonzero((self.name == name) & (self.frame == frame))[0])

  def window(self, name, f0, f1):
    return np.flatnonzero((self.name == name) & (self.frame >= f0) & (self.frame <= f1))


def launches(z):
  """Slow frames (< 4 m/s) after which the car gains >= 3 m/s within 8 s; k = first frame of car accel > 0.3 (held 1 s)."""
  ev = []
  for r in np.unique(z.run):
    m = np.flatnonzero(z.run == r)
    vv, aa, cc = z.v[m], z.ac[m], z.circ[m].sum(1)
    i, n = 0, len(m)
    while i < n - 100:
      if vv[i] >= V_SLOW or vv[i:i + 160].max() < vv[i] + 3:
        i += 1
        continue
      j = i + int(np.argmin(vv[i:i + 160]))
      k = next((q for q in range(j, min(n - 20, j + 200)) if aa[q] > 0.3 and aa[q:q + 20].min() > 0.2), None)
      if k is None or vv[k:k + 200].max() < vv[j] + 3:
        i = j + 1
        continue
      ev.append({'i0': int(m[j]), 'k': int(m[k]), 'vmin': float(vv[j]), 'ctx': float(cc[max(0, k - 200):k + 1].max())})
      q = k
      while q < n and vv[q] < max(5, vv[j] + 3):
        q += 1
      i = q
  return ev


def labels(z):
  """Per frame: 0 other, 1 probe positive, 2 red (wait, approach < 4 m/s, turn on red before the turn), 3 STOP sign,
  4 left out of the fit (unsure), 5 green after the positive window (still slow). Also the open-road mask."""
  lab = np.zeros(z.n, int)
  ev = launches(z)
  greens = [(z.idx(n, fg), z.idx(n, fk)) for n, fg, fk in GREEN]
  for g, _ in greens:
    lab[g - 10:g] = 4  # the real change can be up to 10 frames before the thumbnail
    w = np.arange(g, g + POS_S + 1)
    lab[w[z.v[w] < V_SLOW]] = 1
    q = g + POS_S + 1
    while q < z.n and z.v[q] < V_SLOW:
      lab[q] = 5
      q += 1
  for n, f0, f1 in RED + RED_EXTRA:
    w = z.window(n, f0, f1)
    gs = [g for g, _ in greens if w[0] <= g <= w[-1] + 1]
    ks = [e['k'] for e in ev if w[0] <= e['k'] <= w[-1] + 40]
    cut = min(gs) - 11 if gs else (min(ks) - 40 if ks else w[-1])
    w = w[(w <= cut) & (z.v[w] < V_SLOW)]
    lab[w[lab[w] == 0]] = 2
  for n, fk in RTOR:
    k = z.idx(n, fk)
    w = np.arange(k - 100, k + 1)
    w = w[(z.v[w] < V_SLOW) & (z.run[w] == z.run[k])]
    lab[w[lab[w] == 0]] = 2
  for n, f0, f1 in STOP + STOP_EXTRA:
    w = z.window(n, f0, f1)
    lab[w[np.isin(lab[w], [0, 2, 4])]] = 3
  g_ks = {k for _, k in greens}
  rtor_ks = {z.idx(n, fk) for n, fk in RTOR}
  for e in ev:
    if e['k'] in g_ks:
      continue
    # other launches (lead traffic, no light, unclear) are left out of the fit; after a turn on red too
    w = np.arange(e['k'] - (0 if e['k'] in rtor_ks else 60), min(z.n, e['k'] + 60))
    lab[w[lab[w] == 0]] = 4
  # open road: > 250 m driven before any controlled stop or labeled stop, outside all labels, moving
  far = np.ones(z.n, bool)
  stops = set(np.flatnonzero(z.stop_here).tolist())
  for n, f0, f1 in STOP + STOP_EXTRA + RED + RED_EXTRA:
    w = z.window(n, f0, f1)
    stops.add(int(w[np.argmin(z.v[w])]))
    far[w] = False
  pos = np.zeros(z.n)
  for r in np.unique(z.run):
    m = np.flatnonzero(z.run == r)
    pos[m] = np.concatenate([[0], np.cumsum(z.v[m] * 0.05)])[:-1]
  for s in stops:
    m = np.flatnonzero(z.run == z.run[s])
    m = m[(m <= s) & (pos[s] - pos[m] < 250)]
    far[m] = False
  return lab, far & (lab == 0) & (z.v > 2), greens


def ridge(X, y, wt, lam):
  mu = np.average(X, 0, wt)
  sd = np.sqrt(np.average((X - mu) ** 2, 0, wt)) + 1e-6
  Z = (X - mu) / sd
  ym = np.average(y, weights=wt)
  w = np.linalg.solve((Z * wt[:, None]).T @ Z + lam * wt.sum() * np.eye(Z.shape[1]), (Z * wt[:, None]).T @ (y - ym))
  return w / sd, float(ym - (mu / sd) @ w)


def fit(z, lab, open_road, train):
  every3 = np.arange(z.n) % 3 == 0
  sel = train & ((lab == 1) | (lab == 2) | (lab == 3) | ((lab == 0) & every3))
  wt = np.select([lab[sel] == 1, np.isin(lab[sel], [2, 3])], [20.0, 3.0], 1.0)
  w, b = ridge(z.x[sel].astype(np.float64), (lab[sel] == 1).astype(float), wt, LAM)
  s = z.x @ w + b
  t = max(np.percentile(s[train & (lab == 2)], PCT_NEG), np.percentile(s[train & (lab == 3)], PCT_NEG),
          np.percentile(s[train & open_road], PCT_OPEN)) + MARGIN
  # accel fit on slow frames, from the live r4 units (what a layer-2 unit can read)
  slow = train & (z.v < V_SLOW)
  wa, ba = ridge(z.r4[slow].astype(np.float64), z.raw_s[slow], np.ones(slow.sum()), LAM_A)
  return {'w': w, 'b': b, 't': t, 'wa': wa, 'ba': ba}


def readout(P):
  """Final-layer column for trunk 47: + CAP m/s² of raw accel per unit (long rows only)."""
  tt = np.minimum(T, T_PUSH)
  dv = CAP * tt
  dx = CAP * (tt ** 2 / 2 + T_PUSH * np.maximum(T - T_PUSH, 0))
  da = np.where(T < T_PUSH, CAP, 0.0)
  r = np.zeros(495)
  for k in range(33):
    r[k * 15 + 0], r[k * 15 + 3], r[k * 15 + 6] = dx[k], dv[k], da[k]
  scale = P['scale_layer.plan.scale'][:495]
  out = np.zeros(N_PLAN)
  ok = scale[LONG_ROWS] != 0  # x at t = 0 has scale 0
  out[LONG_ROWS[ok]] = r[LONG_ROWS[ok]] / scale[LONG_ROWS[ok]]
  return out


def circuit(params, P_ship):
  c = Circuit('stop')
  kg = 1.0 / RAMP
  w, b = params['w'] * kg, (params['b'] - params['t']) * kg
  c.r4(SLOT_B, w, b - 1.0)                      # relu(k (s - t) - 1)
  k_clip = c.free('r6')[0]
  c.r6(k_clip, {SLOT_B: 1.0}, 0.0)               # copy of slot 9
  # B = relu(-a_hat / CAP) - relu(-a_hat / CAP - 1), a_hat = wa . r4_live + ba
  wa = {u: -float(v) / CAP for u, v in zip(LIVE_R4, params['wa'], strict=True)}
  k_b1 = c.r6(c.free('r6')[0], wa, -params['ba'] / CAP)
  k_b2 = c.r6(c.free('r6')[0], wa, -params['ba'] / CAP - 1.0)
  # trunk 47 = relu((1 + BETA) G + B - 1), G = r4_47 - r6 copy
  g1 = 1.0 + BETA
  c.out(47, {k_clip: -g1, k_b1: 1.0, k_b2: -1.0}, OVER * readout(P_ship), w_x=w * g1, b_x=b * g1, b=-1.0)
  # the skip cancel writes zeros into dead r4 columns that other patches own; patch_model.py clears them anyway
  own_r4 = {u for kind, u in c.used if kind == 'r4'}
  for key in [key for key, val in c.w.items() if key[0] == 'res_layer.plan.0.weight' and val == 0.0 and
              key[1] % 256 in DEAD_R4 and key[1] % 256 not in own_r4]:
    del c.w[key]
  return c


def rates(da):
  a = np.abs(da)
  if len(a) == 0:
    return 'n=0'
  return (f'n={len(a)}: > 0.1 on {100 * np.mean(a > 0.1):.3f}%, > 0.3 on {100 * np.mean(a > 0.3):.3f}%, '
          f'p99.9 {np.percentile(a, 99.9):.3f}, max {a.max():.2f}')


def first_above(a, g, thr):
  """s from frame g to the first frame (from g - 0.5 s) with accel above thr."""
  return (next(q for q in range(g - 10, g + 200) if a[q] > thr) - g) / 20


def kinematic_v_min(z, a, g):
  """Open-loop proxy: lowest speed if the car had followed accel a from g - 0.5 s for 2.5 s."""
  return np.min(np.maximum(z.v[g - 10] + np.cumsum(a[g - 10:g + 40]) * 0.05, 0.0))


def measure(z, greens):
  """Part 1: how long the master accel stays <= 0 after the green and after the car starts, and the stop units."""
  print(f'GREEN launches (n={len(greens)}, camera-checked), t = 0 at the first green thumbnail (real change 0-0.5 s before):')
  print('   name  frame  v@green | car starts | master a > 0, > 0.3 | master a <= 0 after the car starts | '
        'u201 / u239 / u69 / u248 at -1 s, 0, +0.5 s, +1 s')
  rows = []
  for (n, fg, _), (g, k) in zip(GREEN, greens, strict=True):
    am = z.a_m
    a0 = next(q for q in range(g - 20, g + 200) if am[q] > 0.0)
    a3 = next(q for q in range(g - 20, g + 200) if am[q] > 0.3)
    after = next(q for q in range(k, k + 200) if am[q] > 0.0) - k
    u = ' | '.join(' '.join(f'{x:.2f}' for x in z.circ[g + o, [2, 3, 0, 1]]) for o in (-20, 0, 10, 20))
    print(f'  {n:>8} f{fg:4d} {z.v[g]:4.2f} | {(k - g) / 20:+.2f}s | {(a0 - g) / 20:+.2f}s {(a3 - g) / 20:+.2f}s | {after / 20:.2f}s | {u}')
    rows.append((z.v[g], (k - g) / 20, (a0 - g) / 20, (a3 - g) / 20, after / 20))
  r = np.array(rows)
  creep = r[:, 0] > 0.2
  print(f'  standstill (n={np.sum(~creep)}): car starts median {np.median(r[~creep, 1]):+.2f}s, master a > 0.3 median '
        f'{np.median(r[~creep, 3]):+.2f}s | creeping (n={creep.sum()}): car starts {np.round(r[creep, 1], 2).tolist()}, master a > 0 '
        f'{np.round(r[creep, 2], 2).tolist()}, a <= 0 after the car starts {np.round(r[creep, 4], 2).tolist()} s')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--save', action='store_true')
  args = ap.parse_args()
  if not os.path.exists(CACHE):
    build_cache()
  z = Data()
  lab, open_road, greens = labels(z)
  measure(z, greens)
  P_ship = head.patched_weights(head.SHIPPED)
  ship_plan = head.forward(P_ship, z.x)['plan']
  plan1 = z.plan_s.copy()
  tag = f'm{MARGIN}_r{RAMP}_c{CAP}_b{BETA}_o{OVER}'
  tmp = f'{OUT}/fold_{tag}.npz'
  gate = np.zeros(z.n)
  probe = np.zeros(z.n)  # s - t: the probe is on above 0
  for r in sorted(set(z.route)):
    test = z.route == r
    params = fit(z, lab, open_road, ~test)
    circuit(params, P_ship).save(tmp, 'fold')
    P = head.patched_weights(head.SHIPPED + [tmp])
    f = head.forward(P, z.x[test])
    plan1[test] = z.plan_s[test] + (f['plan'] - ship_plan[test]).astype(np.float32)
    gate[test] = f['trunk'][:, 47]
    probe[test] = z.x[test] @ params['w'] + params['b'] - params['t']
  a1 = np.zeros(z.n)
  for nm in np.unique(z.name):
    m = z.name == nm
    a1[m] = accels(plan1[m])
  da = a1 - z.a_s
  np.savez(f'{OUT}/loro_{tag}.npz', a1=a1, gate=gate, probe=probe)
  print(f'\ncircuit {tag}, held out by route. Change in accel (shipped -> shipped + circuit):')
  print('  GREEN events (t = 0 at the green thumbnail): probe on at (car starts at) | shipped -> edited: a > 0 at | a > 0.3 at | '
        'max extra accel in [-0.5 s, +2 s] | kinematic: lowest speed if the car followed the accel from -0.5 s for 2.5 s')
  for (n, fg, _), (g, k) in zip(GREEN, greens, strict=True):
    on = next((q for q in range(g - 40, g + 60) if probe[q] > 0), None)
    on = '  -  ' if on is None else f'{(on - g) / 20:+.2f}s'
    print(f'    {n:>8} f{fg:4d} v{z.v[g]:.2f}: {on} ({(k - g) / 20:+.2f}s) | '
          f'{first_above(z.a_s, g, 0):+.2f}s -> {first_above(a1, g, 0):+.2f}s | '
          f'{first_above(z.a_s, g, 0.3):+.2f}s -> {first_above(a1, g, 0.3):+.2f}s | {da[g - 10:g + 40].max():+.2f} | '
          f'{kinematic_v_min(z, z.a_s, g):.2f} -> {kinematic_v_min(z, a1, g):.2f} m/s')
  red = lab == 2
  checks = {
    'red standstill (v < 0.3)': da[red & (z.v < 0.3)],
    'red, slow moving (0.3-4 m/s)': da[red & (z.v >= 0.3)],
    'STOP approach + standstill': da[lab == 3],
    'open road (> 250 m from a stop, v > 2)': da[open_road],
    'all other frames': da[(lab == 0) & ~open_road],
    'unsure / other launches (not fit)': da[lab == 4],
  }
  for k, v in checks.items():
    print(f'  {k:40s} {rates(v)}')
  ok = (np.mean(np.abs(checks['red standstill (v < 0.3)']) > 0.1) <= 0.005 and
        np.mean(np.abs(checks['red, slow moving (0.3-4 m/s)']) > 0.1) <= 0.005 and
        np.mean(np.abs(checks['STOP approach + standstill']) > 0.1) <= 0.005 and
        np.mean(np.abs(checks['open road (> 250 m from a stop, v > 2)']) > 0.1) <= 0.001)
  print('PASS' if ok else 'FAIL', '(limits: red and STOP > 0.1 on <= 0.5% of frames, open road <= 0.1%)')
  if args.save:
    assert ok, 'the held-out checks fail: nothing saved'
    params = fit(z, lab, open_road, np.ones(z.n, bool))
    out = f'{CODE}/patches/green_go_v1.npz'
    circuit(params, P_ship).save(out, f'green after creeping: cancel braking (cap {CAP}) when a green probe is on, trunk 47')
    head.patched_weights(head.SHIPPED + [out])  # raises on a clash with the shipped patches


if __name__ == '__main__':
  main()
