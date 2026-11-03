#!/usr/bin/env python3
"""List disengagements and gas/steer overrides on a route with the model and car state around them.

usage: disengages.py ROUTE
"""
import sys
from multiprocessing import Pool
import numpy as np
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route


def scan(seg):
  rows = []
  st = {'en': None, 'gas': False, 'brake': False, 'steer': False, 'v': 0.0, 'a': 0.0, 'curv': 0.0, 'model_a': 0.0,
        'lead_x': np.nan, 'lead_v': np.nan, 'src': '', 'big': None, 'events': [], 'vcruise': 0.0, 'btn': ''}
  t0 = None
  for m in LogReader(seg, sort_by_time=True):
    w = m.which()
    if t0 is None:
      t0 = m.logMonoTime
    if w == 'carState':
      cs = m.carState
      for k, v in (('gas', cs.gasPressed), ('brake', cs.brakePressed), ('steer', cs.steeringPressed)):
        if v and not st[k] and st['en']:
          rows.append((m.logMonoTime, f'{k}_press', dict(st)))
        st[k] = v
      st['v'], st['a'], st['vcruise'] = cs.vEgo, cs.aEgo, cs.vCruise
      if len(cs.buttonEvents):
        st['btn'] = ','.join(str(b.type) for b in cs.buttonEvents if b.pressed)
    elif w == 'selfdriveState':
      en = m.selfdriveState.enabled
      if st['en'] and not en:
        rows.append((m.logMonoTime, 'DISENGAGE', dict(st)))
      st['en'] = en
    elif w == 'onroadEvents':
      st['events'] = [str(e.name) for e in m.onroadEvents]
    elif w == 'modelV2':
      mv = m.modelV2
      st['model_a'], st['curv'], st['big'] = mv.action.desiredAcceleration, mv.action.desiredCurvature, mv.big
      l = mv.leadsV3[0]
      st['lead_x'], st['lead_v'] = (l.x[0], l.v[0]) if l.prob > 0.5 else (np.nan, np.nan)
    elif w == 'longitudinalPlan':
      st['src'] = str(m.longitudinalPlan.longitudinalPlanSource)
  return seg, rows


if __name__ == '__main__':
  route = sys.argv[1]
  segs = [f'{route}/{s.name.segment_num}' for s in Route(route).segments if s.log_path]
  with Pool(8) as p:
    res = sorted(p.map(scan, segs), key=lambda r: int(r[0].rsplit('/', 1)[1]))
  start = None
  for seg, rows in res:
    for t, kind, s in rows:
      if start is None:
        start = t - int(seg.rsplit('/', 1)[1]) * 60 * 1e9  # approximate route start
      rt = (t - start) / 1e9
      print(f"{rt:7.1f}s {kind:<10} v={s['v']:5.1f} a={s['a']:+.2f} model_a={s['model_a']:+.2f} src={s['src']:<7} "
            f"curv={s['curv']:+.4f} lead=({s['lead_x']:.0f}m,{s['lead_v']:.1f}) vcruise={s['vcruise']:.0f} big={s['big']} "
            f"gas={s['gas']} brk={s['brake']} steer={s['steer']} ev={s['events'][:3]} btn={s['btn']}")
