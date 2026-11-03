#!/usr/bin/env python3
"""Cache for the 2-layer corner-cut circuit: per-frame inputs and plans of the 111 base segments.

Writes $WEIGHT_LAB_DATA/exp/cut/cache.npz with x_plan_feat, speed, lanes, the master plan and the plan of
the shipped patch stack (head.SHIPPED). All later fits and scores read only this file.

usage: exp_cut_data.py
"""
import os
import numpy as np

from openpilot.tools.weight_lab import head
from openpilot.tools.weight_lab import fit_lat_edit as F
from openpilot.tools.weight_lab.lab_paths import DATA

OUT_DIR = f'{DATA}/exp/cut'
CACHE = f'{OUT_DIR}/cache.npz'


def build():
  master = head.master_weights()
  shipped = head.patched_weights(head.SHIPPED)
  cols = {k: [] for k in ('x', 'v', 'curv', 'center', 'lane_prob', 'lc_prob', 'plan', 'plan_ship', 'route', 'seg', 'frame')}
  for p in head.base_files():
    d = np.load(p)
    center, prob = F.lane_center_all(d['raw'])
    ref = head.forward(master, d['x_plan_feat'])
    cols['x'].append(d['x_plan_feat'])
    cols['v'].append(d['v_ego'])
    cols['curv'].append(d['curv'])
    cols['center'].append(center.astype(np.float32))
    cols['lane_prob'].append(prob.astype(np.float32))
    cols['lc_prob'].append(F.lane_change_prob(d['raw']).astype(np.float32))
    cols['plan'].append(d['plan'])
    cols['plan_ship'].append(head.edited_plan(d, shipped, ref))
    cols['route'].append(np.full(len(d['v_ego']), str(d['seg'][0]).rsplit('/', 1)[0]))
    cols['seg'].append(np.full(len(d['v_ego']), os.path.basename(p)[:-4]))
    cols['frame'].append(d['frame'])
    print(os.path.basename(p), len(d['v_ego']))
  os.makedirs(OUT_DIR, exist_ok=True)
  np.savez(CACHE, **{k: np.concatenate(v) for k, v in cols.items()})
  print(f'wrote {CACHE}')


def load():
  z = np.load(CACHE)
  return {k: z[k] for k in z.files}


if __name__ == '__main__':
  build()
