"""Exact numpy plan head: score any stack of patches on the saved activations, no replay.

The plan head reads x_plan_feat and nothing feeds back into the model state, so
  edited plan = replayed plan + head(patched weights, x) - head(master weights, x).

  P = patched_weights(['patches/stop_gain_1p8.npz', 'my.npz'])
  plan = edited_plan(d, P)          # d = np.load('seg/fe_2.npz'), plan (n, 33, 15) like d['plan']
"""
import glob
import json
import os
import numpy as np

from openpilot.tools.weight_lab.lab_paths import CODE, DATA
from openpilot.tools.weight_lab.slots import H, DEAD_R4, DEAD_R6

BASE_PREFIXES = ('fe', 'fd', 'f7', 'id4', 'n76', 'n77', 'b2d', 'be', 's39', 's606', 's608')
SHIPPED = [f'{CODE}/patches/{p}' for p in ('stop_gain_u201_1p4.npz', 'lat_readout_wlo20_half.npz')]  # same list as build_model.sh
NAMES = ('in_layer.plan.weight', 'in_layer.plan.bias', 'res_layer.plan.0.weight', 'res_layer.plan.0.bias',
         'res_layer.plan.2.weight', 'res_layer.plan.2.bias', 'final_layer.plan.weight', 'final_layer.plan.bias',
         'scale_layer.plan.scale')


def master_weights():
  z = np.load(f'{DATA}/head_weights.npz')
  return {n: z[H + n].astype(np.float64) for n in NAMES}


def clear_dead(P):
  """The same clearing as patch_model.py: dead units can only feed circuits that own them."""
  P['res_layer.plan.0.weight'][:, list(DEAD_R4)] = 0.0
  P['res_layer.plan.2.weight'][:, list(DEAD_R6)] = 0.0
  return P


def load_patch(path):
  z = np.load(path)
  meta = json.loads(str(z['meta']))
  return meta, [(e['name'], z[f'idx{i}'], z[f'val{i}']) for i, e in enumerate(meta['edits'])]


def patched_weights(patches, base=None):
  """Master weights, cleared, with the patches applied in order. Raises on two patches writing one weight."""
  P = clear_dead({k: v.copy() for k, v in (base or master_weights()).items()})
  touched = {}
  for p in patches:
    for name, idx, val in load_patch(p)[1]:
      n = name.replace(H, '')
      prev = touched.setdefault(n, {})
      clash = set(idx.tolist()) & set(prev)
      assert not clash, f'{p} and {prev[next(iter(clash))]} both write {len(clash)} weights of {n}'
      prev.update(dict.fromkeys(idx.tolist(), p))
      P[n].reshape(-1)[idx] = val
  return P


def forward(P, x):
  """x (n, 512) -> dict of r4, r6, trunk_pre, trunk, plan mu (n, 33, 15) incl. scale."""
  x = np.asarray(x, dtype=np.float64)
  r4 = np.maximum(x @ P['in_layer.plan.weight'].T + P['in_layer.plan.bias'], 0)
  r6 = np.maximum(r4 @ P['res_layer.plan.0.weight'].T + P['res_layer.plan.0.bias'], 0)
  pre = r4 + r6 @ P['res_layer.plan.2.weight'].T + P['res_layer.plan.2.bias']
  trunk = np.maximum(pre, 0)
  s = P['scale_layer.plan.scale'][:495]
  mu = (trunk @ P['final_layer.plan.weight'][:495].T + P['final_layer.plan.bias'][:495]) * s
  return {'r4': r4, 'r6': r6, 'trunk_pre': pre, 'trunk': trunk, 'plan': mu.reshape(-1, 33, 15)}


_MASTER = None


def edited_plan(d, P, ref=None):
  """Replayed plan plus the exact change from master to P. ref: master forward() of d, if already computed."""
  global _MASTER
  if ref is None:
    _MASTER = _MASTER or master_weights()
    ref = forward(_MASTER, d['x_plan_feat'])
  return d['plan'] + (forward(P, d['x_plan_feat'])['plan'] - ref['plan']).astype(np.float32)


def base_files():
  return [p for p in sorted(glob.glob(f'{DATA}/seg/*.npz')) if os.path.basename(p).split('_')[0] in BASE_PREFIXES]
