#!/usr/bin/env python3
"""Build data for viewer/lat.html: model paths and lanes projected onto the thumbnails.

Paths: online (logged modelV2), offline (small model replay), and any edited replays found as
seg/<EDIT>_<prefix>_<n>.npz or passed in EDITS. Projection matches the openpilot UI.

usage: build_lat.py PREFIX:ROUTE:SEG [...]   e.g. n76:f73c01590368ee5b/00000076--cbf1d60cd3:4
"""
import glob
import json
import os
import sys
import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.tools.lib.logreader import LogReader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cut import inside_offset  # noqa: E402

from openpilot.tools.weight_lab.lab_paths import CODE, DATA as HERE  # noqa: F401
OUT = f'{HERE}/viewer'
X_IDXS = np.array([192.0 * (i / 32) ** 2 for i in range(33)])
THUMB_SCALE = 0.25
MAX_X = 80.0


def read_log(seg):
  calib, height, online, dev, sensor = [], [], {}, None, None
  enc = {}
  for m in LogReader(seg, sort_by_time=True):
    w = m.which()
    if w == 'extrinsicsCalibration':
      calib.append((m.logMonoTime, np.array(m.extrinsicsCalibration.rpyCalib), m.extrinsicsCalibration.height[0] if len(m.extrinsicsCalibration.height) else 1.22))
    elif w == 'narrowRoadEncodeIdx':
      enc[m.narrowRoadEncodeIdx.frameId] = m.narrowRoadEncodeIdx.segmentId
    elif w == 'modelV2':
      mv = m.modelV2
      fi = enc.get(mv.frameId)
      if fi is not None:
        online[fi] = (m.logMonoTime, np.array([mv.position.x, mv.position.y, mv.position.z]).T, bool(mv.big))
    elif w == 'deviceState' and dev is None:
      dev = str(m.deviceState.deviceType)
    elif w == 'narrowRoadCameraState' and sensor is None:
      sensor = str(m.narrowRoadCameraState.sensor)
  return calib, online, DEVICE_CAMERAS[(dev, sensor)].narrow_road.intrinsics


def projector(intrinsics, rpy, height):
  T = intrinsics @ view_frame_from_device_frame @ rot_from_euler(rpy)

  def proj(pts, z_offset):
    pts = pts[(pts[:, 0] > 2) & (pts[:, 0] < MAX_X)]
    if len(pts) == 0:
      return []
    p = (T @ (pts + np.array([0, 0, z_offset])).T).T
    p = p[p[:, 2] > 0]
    return np.round(p[:, :2] / p[:, 2:3] * THUMB_SCALE, 1).tolist()
  return proj


def lanes_from_raw(raw):
  lanes = raw[117:645][:264].reshape(4, 33, 2)
  prob = 1 / (1 + np.exp(-raw[645:653].reshape(4, 2)[:, 1]))
  return [np.stack([X_IDXS, lanes[i, :, 0], lanes[i, :, 1]], axis=1) for i in range(4)], prob


def build(prefix, route, num):
  key = f"{route.replace('/', '_')}_{num}"
  seg = f'{route}/{num}'
  base = np.load(f'{HERE}/seg/{prefix}_{num}.npz')
  edits = {os.path.basename(p).split('_')[0]: np.load(p) for p in glob.glob(f'{HERE}/seg/cut*_{prefix}_{num}.npz')}
  calib, online, intr = read_log(seg)
  thumbs = sorted(int(os.path.splitext(os.path.basename(p))[0]) for p in glob.glob(f'{OUT}/thumbs/{key}/*.jpg'))
  fr = base['frame']
  ins, _, lprob, _ = inside_offset(base)
  # the inside offset is only meaningful where the lanes are confident: blank it elsewhere
  conf = lprob > 0.5
  ins = np.where(conf[:, None], ins, np.nan)
  ins_edit = {k: [None if not c else round(float(x), 3) for x, c in zip(inside_offset(e)[0][:, 1], conf)] for k, e in edits.items()}
  frames = {}
  ci = 0
  for f in thumbs:
    i = int(np.argmin(np.abs(fr - f)))
    t = base['t'][i]
    while ci + 1 < len(calib) and calib[ci + 1][0] <= t:
      ci += 1
    if not calib:
      continue
    proj = projector(intr, calib[ci][1], calib[ci][2])
    h = calib[ci][2]
    lanes, lp = lanes_from_raw(base['raw'][i])
    item = {'offline': proj(base['plan'][i, :, 0:3], h), 'lanes': [proj(l, 0.0) for l in lanes], 'lane_prob': np.round(lp, 2).tolist()}
    if f in online:
      item['online'] = proj(online[f][1], h)
      item['online_big'] = online[f][2]
    for k, e in edits.items():
      j = int(np.argmin(np.abs(e['frame'] - f)))
      item[f'edit_{k}'] = proj(e['plan'][j, :, 0:3], h)
    frames[f] = item
  r = lambda a, n=3: [None if not np.isfinite(x) else round(float(x), n) for x in a]
  js = {'seg': seg, 'key': key, 'frame': fr.tolist(), 'thumbs': thumbs, 'frames': frames, 'edits': sorted(edits),
        'series': {'inside20': r(ins[:, 1]), 'lat_acc': r(base['curv'] * base['v_ego'] ** 2, 2), 'v_ego': r(base['v_ego'], 2),
                   'lane_prob': r(lprob, 2), **{f'inside20_{k}': v for k, v in ins_edit.items()}}}
  os.makedirs(f'{OUT}/lat', exist_ok=True)
  with open(f'{OUT}/lat/{key}.json', 'w') as f:
    json.dump(js, f, separators=(',', ':'))
  return {'key': key, 'seg': seg, 'base': base, 'edits': edits}


def clip_metrics(b, clip, num):
  """Mean |inside offset| @20 m on confident-lane frames inside the clip window, offline and per edit."""
  base, edits = b['base'], b['edits']
  f0, f1 = (clip['start'] - 60 * num) * 20, (clip['end'] - 60 * num) * 20
  out = {}
  for name, d in [('offline', base)] + list(edits.items()):
    ins, _, prob, _ = inside_offset(d)
    m = (d['frame'] >= f0) & (d['frame'] <= f1) & (prob > 0.5)
    out[name] = None if m.sum() < 5 else round(float(np.abs(ins[m, 1]).mean()), 2)
  out['confident_frames'] = int(((base['frame'] >= f0) & (base['frame'] <= f1) & (inside_offset(base)[2] > 0.5)).sum())
  return out


def main():
  clips = json.load(open(f'{CODE}/clips.json'))
  built, segs, out_clips = {}, [], []
  for c in clips:
    # the clip lives in the segment that holds most of it
    num = int(((c['start'] + c['end']) / 2) // 60)
    if not os.path.exists(f"{HERE}/seg/{c['prefix']}_{num}.npz"):
      print('skip (not replayed yet):', c['route'], num, c['label'])
      continue
    k = (c['route'], num)
    if k not in built:
      built[k] = build(c['prefix'], c['route'], num)
      segs.append({'key': built[k]['key'], 'seg': built[k]['seg']})
      print('built', built[k]['seg'])
    out_clips.append(c | {'key': built[k]['key'], 'num': num, 'f0': (c['start'] - 60 * num) * 20, 'f1': (c['end'] - 60 * num) * 20,
                          'metrics': clip_metrics(built[k], c, num)})
  json.dump({'segs': segs, 'clips': out_clips}, open(f'{OUT}/lat/index.json', 'w'))


if __name__ == '__main__':
  main()
