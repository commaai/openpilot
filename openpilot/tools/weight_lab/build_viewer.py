#!/usr/bin/env python3
"""Build the local weight-lab viewer from replay npz files and findings.json.

usage: build_viewer.py   (reads seg/*.npz, writes viewer/index.html + viewer/data/*.json + heatmaps)
"""
import glob
import json
import os
import numpy as np
from PIL import Image

from openpilot.tools.weight_lab.lab_paths import CODE, DATA as HERE  # noqa: F401
OUT = f'{HERE}/viewer'
CIRCUIT = [69, 248, 201, 239]
LEAD = slice(917, 1061)
LEAD_PROB = slice(1061, 1064)


def seg_key(seg):
  route, num = seg.rsplit('/', 1)
  return f"{route.replace('/', '_')}_{num}"


def heatmap(h, path):
  """Units x time image. Units are ordered by when they first peak, so circuits form diagonal bands."""
  z = h / (np.percentile(h, 99.5, axis=0) + 1e-6)
  z = np.clip(z, 0, 1)
  order = np.lexsort((np.argmax(h, axis=0), (h > 1e-6).mean(0) > 0.5))
  img = (255 * (1 - z[:, order].T)).astype(np.uint8)  # dark = active
  Image.fromarray(img).resize((h.shape[0], 256)).save(path)
  return order.tolist()


def build_seg(path, modes):
  d = np.load(path)
  seg = str(d['seg'][0])
  key = seg_key(seg)
  h = d['x_plan_trunk']
  z_long = h @ modes['Vt'].T
  z_lat = h @ modes['Vt_lat'].T
  lead_raw = d['raw'][:, LEAD][:, :72].reshape(-1, 3, 6, 4)
  lead_prob = 1 / (1 + np.exp(-d['raw'][:, LEAD_PROB][:, 0]))
  radar = d['radar'] if 'radar' in d.files else np.full((len(h), 5), np.nan)
  thumbs = sorted(int(os.path.splitext(os.path.basename(p))[0]) for p in glob.glob(f'{OUT}/thumbs/{key}/*.jpg'))
  order = heatmap(h, f'{OUT}/data/{key}_trunk.png')
  r = lambda a, n=3: [None if not np.isfinite(x) else round(float(x), n) for x in a]
  js = {
    'seg': seg, 'key': key, 'frame': d['frame'].tolist(), 'thumbs': thumbs, 'unit_order': order,
    'series': {
      'v_ego': r(d['v_ego'], 2), 'accel': r(d['accel']), 'curv': r(d['curv'], 4),
      **{f'u{u}': r(h[:, u]) for u in CIRCUIT},
      'circuit': r(h[:, CIRCUIT].sum(1)),
      'long0': r(z_long[:, 0]), 'long1': r(z_long[:, 1]), 'long2': r(z_long[:, 2]),
      'lat0': r(z_lat[:, 0]), 'lat1': r(z_lat[:, 1]),
      'lead_prob': r(lead_prob), 'lead_x': r(np.where(lead_prob > .5, lead_raw[:, 0, 0, 0], np.nan), 1),
      'lead_v': r(np.where(lead_prob > .5, lead_raw[:, 0, 0, 2], np.nan), 2),
      'radar_v': r(radar[:, 2], 2), 'radar_x': r(radar[:, 1], 1),
    },
  }
  with open(f'{OUT}/data/{key}.json', 'w') as f:
    json.dump(js, f, separators=(',', ':'))
  return {'key': key, 'seg': seg, 'frames': len(h)}


def main():
  os.makedirs(f'{OUT}/data', exist_ok=True)
  modes = np.load(f'{HERE}/plan_modes.npz')
  segs = [build_seg(p, modes) for p in sorted(glob.glob(f'{HERE}/seg/*.npz'))]
  findings = json.load(open(f'{CODE}/findings.json')) if os.path.exists(f'{CODE}/findings.json') else []
  with open(f'{OUT}/data/index.json', 'w') as f:
    json.dump({'segs': segs, 'findings': findings, 'circuit': CIRCUIT}, f)
  print(f'{len(segs)} segments, {len(findings)} findings')


if __name__ == '__main__':
  main()
