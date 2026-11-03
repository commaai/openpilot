#!/usr/bin/env python3
"""Lead circuit: teach the e2e plan to brake for closing leads, with the lead MPC as the teacher.

Target per frame: the extra braking the MPC asked for, m = clip(e2e - mpc, 0, M_MAX) when the MPC was limiting
(longitudinalPlanSource lead*), else 0. e2e = the shipped patched model accel.

Circuit (owner 'lead', slots.py):
  r6 pairs:  h_i = relu(u_i) - relu(u_i - 1) = clamp(u_i, 0, 1),  u_i = w_i . r4 + b_i   (r4 = live first layer)
  trunk 227: out = relu(sum v_i h_i + b), bounded by sum |v_i|
  readout:   constant decel profile for 0-3 s (ax -1, vx -t, x -t^2/2), so 1 unit of out = -1 m/s² of accel.

usage: exp_lead.py cache        read the logs once (lead MPC target, radar lead) into exp/lead/cache.npz
       exp_lead.py fit          fit on 2/3 of the routes, score on the rest (3 folds), then fit on all
       exp_lead.py save OUT     write the patch fitted on all routes
"""
import os
import sys
from multiprocessing import Pool
import numpy as np
import torch

from openpilot.tools.weight_lab import head
from openpilot.tools.weight_lab.lab_paths import DATA
from openpilot.tools.weight_lab.lead_vs_e2e import scan, LEAD_SOURCES
from openpilot.tools.weight_lab.lights_edit import accels
from openpilot.tools.weight_lab.slots import Circuit, OWN, LIVE_R4, N_PLAN

OUT_DIR = f'{DATA}/exp/lead'
T = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
M_MAX = 2.5        # m/s², largest extra braking the circuit learns
N_FEAT = 11        # clamped features, 2 r6 units each
BRAKE_T = 3.0      # s, length of the decel profile in the readout
FP_WEIGHT = 4.0    # braking where the MPC did not ask for it costs this much more


def readout(scale):
  """Plan change for -1 m/s² over 0-BRAKE_T s, in raw output units (before the scale layer)."""
  tb = np.minimum(T, BRAKE_T)
  d = np.zeros((33, 15))
  d[:, 6] = -(T < BRAKE_T).astype(float)
  d[:, 3] = -tb
  d[:, 0] = -(tb ** 2 / 2 + tb * (T - tb))
  col = np.zeros(N_PLAN)
  s = scale[:495]
  col[:495] = np.where(s != 0, d.reshape(-1) / np.where(s != 0, s, 1), 0)
  return col


def cache():
  os.makedirs(OUT_DIR, exist_ok=True)
  files = head.base_files()
  with Pool(12) as pool:
    res = pool.map(scan, files)
  P = head.patched_weights(head.SHIPPED)
  out = {}
  for f, r in zip(files, res, strict=True):
    if r is None:
      continue
    d = np.load(f)
    key = os.path.basename(f)[:-4]
    out[key + '/src_lead'] = np.isin(r['src'], LEAD_SOURCES)
    out[key + '/mpc'] = r['mpc']
    out[key + '/lead'] = r['lead']
    out[key + '/e2e'] = accels(head.edited_plan(d, P)).astype(np.float32)
  np.savez(f'{OUT_DIR}/cache.npz', **out)
  print(f'{len(out) // 4} segments cached')


def load():
  c = np.load(f'{OUT_DIR}/cache.npz')
  M = head.master_weights()
  P = head.patched_weights(head.SHIPPED)
  segs = []
  for key in sorted({k.split('/')[0] for k in c.files}):
    d = np.load(f'{DATA}/seg/{key}.npz')
    r4 = head.forward(M, d['x_plan_feat'])['r4'][:, LIVE_R4]
    src, mpc, e2e, v = c[key + '/src_lead'], c[key + '/mpc'], c[key + '/e2e'], d['v_ego']
    m = np.where(src & (v > 3), np.clip(e2e - mpc, 0, M_MAX), 0.0)
    segs.append(dict(key=key, route=key.split('_')[0], d=d, P=P, r4=r4, m=m, src=src & (v > 3), mpc=mpc, e2e=e2e,
                     lead=c[key + '/lead'], v=v))
  return segs


class Net(torch.nn.Module):
  def __init__(self, n_in):
    super().__init__()
    self.l1 = torch.nn.Linear(n_in, N_FEAT)
    self.v = torch.nn.Parameter(torch.full((N_FEAT,), 0.1))
    self.b = torch.nn.Parameter(torch.tensor(-0.5))

  def forward(self, x):
    h = torch.clamp(self.l1(x), 0, 1)
    return torch.relu(h @ self.v + self.b)


def fit(segs, seed=0):
  torch.manual_seed(seed)
  X = torch.tensor(np.concatenate([s['r4'] for s in segs]), dtype=torch.float32, device='cuda')
  y = torch.tensor(np.concatenate([s['m'] for s in segs]), dtype=torch.float32, device='cuda')
  mu, sd = X.mean(0), X.std(0) + 1e-3
  net = Net(X.shape[1]).cuda()
  opt = torch.optim.Adam(net.parameters(), lr=3e-3)
  pos = (y > 0).float()
  w = 1.0 + 20.0 * pos  # positives are rare (a few %)
  for _ in range(3000):
    p = net((X - mu) / sd)
    err = p - y
    loss = (w * torch.where(err > 0, FP_WEIGHT * err ** 2, err ** 2)).mean() + 1e-4 * net.v.abs().sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
  # fold the input normalization into the first layer
  W = (net.l1.weight / sd).detach().cpu().double().numpy()
  b = (net.l1.bias - (net.l1.weight / sd) @ mu).detach().cpu().double().numpy()
  return dict(W=W, b=b, v=net.v.detach().cpu().double().numpy(), b_out=float(net.b.detach()))


def build(c):
  P = head.master_weights()
  circ = Circuit('lead')
  r6 = circ.free('r6')
  w6 = {}
  for i in range(N_FEAT):
    ka, kb = r6[2 * i], r6[2 * i + 1]
    w = dict(zip(LIVE_R4, c['W'][i], strict=True))
    circ.r6(ka, w, c['b'][i])
    circ.r6(kb, w, c['b'][i] - 1.0)
    w6[ka], w6[kb] = c['v'][i], -c['v'][i]
  circ.out(OWN['lead']['trunk'][0], w6, readout(P['scale_layer.plan.scale']), b=c['b_out'])
  return circ


def score(segs, patch):
  """Compare e2e with the circuit to the MPC: episodes, extra braking where the MPC did not ask, tails."""
  P = head.patched_weights(head.SHIPPED + [patch])
  n_ep0 = n_ep1 = 0
  da_neg, da_pos, worst = [], [], []
  for s in segs:
    a1 = accels(head.edited_plan(s['d'], P))
    da = a1 - s['e2e']
    for a, k in ((s['e2e'], 0), (a1, 1)):
      bad = s['src'] & (a - s['mpc'] > 1.0)
      on = np.flatnonzero(bad)
      n = sum(len(e) >= 10 for e in np.split(on, np.flatnonzero(np.diff(on) > 5) + 1)) if len(on) else 0
      if k == 0:
        n_ep0 += n
      else:
        n_ep1 += n
    need = s['m'] > 0.3
    da_pos.append((da[need], s['m'][need]))
    quiet = ~s['src'] & (s['v'] > 3)
    da_neg.append(da[quiet])
    i = np.argmin(np.where(quiet, da, 0))
    worst.append((da[i], s['key'], int(s['d']['frame'][i]), float(s['v'][i]), s['lead'][i].tolist()))
  q = np.concatenate(da_neg)
  dp = np.concatenate([a for a, _ in da_pos])
  mp = np.concatenate([b for _, b in da_pos])
  worst.sort()
  print(f'  under-braking episodes (e2e - mpc > 1 for >= 0.5 s): {n_ep0} -> {n_ep1}')
  print(f'  frames the MPC wanted > 0.3 m/s² more ({len(dp)}): median extra braking {-np.median(dp):.2f} of median {np.median(mp):.2f} needed')
  print(f'  MPC not limiting ({len(q)} frames): d accel < -0.3 on {100 * np.mean(q < -0.3):.2f}%, < -0.1 on {100 * np.mean(q < -0.1):.2f}%, '
        f'p99.9 {np.percentile(q, 0.1):+.2f}, min {q.min():+.2f}')
  for w in worst[:5]:
    print(f'    worst quiet: {w[0]:+.2f} {w[1]} f{w[2]} v {w[3]:.1f} lead(present,d,vrel,radar) {np.round(w[4], 1).tolist()}')


def main():
  cmd = sys.argv[1]
  if cmd == 'cache':
    return cache()
  segs = load()
  routes = sorted({s['route'] for s in segs})
  if cmd == 'fit':
    for fold in range(3):
      test = set(routes[fold::3])
      c = fit([s for s in segs if s['route'] not in test])
      circ = build(c)
      path = f'{OUT_DIR}/fold{fold}.npz'
      circ.save(path, f'lead circuit fold {fold}')
      print(f'fold {fold}, held-out routes {sorted(test)}:')
      score([s for s in segs if s['route'] in test], path)
  elif cmd == 'save':
    circ = build(fit(segs))
    circ.save(sys.argv[2], 'lead circuit: extra braking for closing leads, taught by the lead MPC')
    print('all routes (in-sample):')
    score(segs, sys.argv[2])


if __name__ == '__main__':
  main()
