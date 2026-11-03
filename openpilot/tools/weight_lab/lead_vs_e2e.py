#!/usr/bin/env python3
"""Would e2e long alone brake enough for leads? Compare the logged lead-MPC target to the small model's e2e accel.

For each replayed segment: frames where the logged plan source was a lead (MPC limiting), the MPC aTarget,
the replayed small-model accel, and the lead gap / closing speed from radarState. Under-braking episode:
e2e - mpc > 1 m/s² for 0.5 s or more. TTC = dRel / closing speed.

usage: lead_vs_e2e.py
"""
import glob
import os
from multiprocessing import Pool
import numpy as np

from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.weight_lab.lab_paths import DATA
from openpilot.tools.weight_lab.lights_edit import BASE_PREFIXES

LEAD_SOURCES = ('lead0', 'lead1', 'lead2')


def scan(p):
  try:
    d = np.load(p)
    seg = str(d['seg'][0])
    t = d['t']
  except Exception:
    return None
  lp_t, lp_src, lp_a, rs_t, rs = [], [], [], [], []
  for m in LogReader(seg, sort_by_time=True):
    w = m.which()
    if w == 'longitudinalPlan':
      lp_t.append(m.logMonoTime); lp_src.append(str(m.longitudinalPlan.longitudinalPlanSource)); lp_a.append(m.longitudinalPlan.aTarget)
    elif w == 'radarState':
      l = m.radarState.leadOne
      rs_t.append(m.logMonoTime); rs.append((l.present, l.dRel, l.vRel, l.radar))
  if not lp_t or not rs_t:
    return None
  i = np.clip(np.searchsorted(lp_t, t), 0, len(lp_t) - 1)
  j = np.clip(np.searchsorted(rs_t, t), 0, len(rs_t) - 1)
  rs = np.array(rs, dtype=np.float64)[j]
  return dict(seg=seg, v=d['v_ego'], e2e=d['accel'], src=np.array(lp_src)[i], mpc=np.array(lp_a)[i], lead=rs, frame=d['frame'])


def main():
  files = [p for p in sorted(glob.glob(f'{DATA}/seg/*.npz')) if os.path.basename(p).split('_')[0] in BASE_PREFIXES]
  with Pool(12) as pool:
    res = [r for r in pool.map(scan, files) if r is not None]
  n_lead = n_all = 0
  diffs, eps = [], []
  for r in res:
    lead_lim = np.isin(r['src'], LEAD_SOURCES) & (r['v'] > 3)
    n_lead += lead_lim.sum(); n_all += (r['v'] > 3).sum()
    dd = r['e2e'] - r['mpc']
    diffs.append(dd[lead_lim])
    bad = lead_lim & (dd > 1.0)
    on = np.flatnonzero(bad)
    if len(on) == 0:
      continue
    for e in np.split(on, np.flatnonzero(np.diff(on) > 5) + 1):
      if len(e) < 10:
        continue
      k = e[np.argmax(dd[e])]
      d_rel, v_rel = r['lead'][k, 1], r['lead'][k, 2]
      ttc = d_rel / -v_rel if v_rel < -0.1 else np.inf
      eps.append((r['seg'], int(r['frame'][e[0]]), len(e) / 20, float(r['mpc'][k]), float(r['e2e'][k]), float(r['v'][k]), d_rel, v_rel, ttc, bool(r['lead'][k, 3])))
  D = np.concatenate(diffs)
  print(f'{len(res)} segments; lead-MPC limiting on {100 * n_lead / max(n_all, 1):.1f}% of moving frames')
  print(f'on those frames, e2e - mpc: median {np.median(D):+.2f}, p90 {np.percentile(D, 90):+.2f}, p99 {np.percentile(D, 99):+.2f} m/s²; '
        f'e2e brakes >1 m/s² less than mpc on {100 * np.mean(D > 1):.1f}%')
  eps.sort(key=lambda e: e[8])
  print(f'{len(eps)} under-braking episodes (e2e - mpc > 1 for >= 0.5 s), closest TTC first:')
  for e in eps[:20]:
    print(f'  {e[0]} f{e[1]} {e[2]:.1f}s: mpc {e[3]:+.2f} e2e {e[4]:+.2f} | v {e[5]:.1f} lead {e[6]:.0f} m vRel {e[7]:+.1f} TTC {e[8]:.1f}s radar {e[9]}')


if __name__ == '__main__':
  main()
