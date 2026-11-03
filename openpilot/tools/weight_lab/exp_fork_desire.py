#!/usr/bin/env python3
"""Replay one segment with a forced turn desire, to see if the plan commits to a turn.

usage: exp_fork_desire.py ROUTE/SEG --desire left|right|none --start F [--every N] [--stop F] --out out.npz [--onnx M]
  --start F   frame of the first desire pulse
  --every N   send a new pulse each N frames after that (0: one pulse only)
  --stop F    stop the replay after this frame, to save decode time

Same loop as replay.replay_segment, but the desire input comes from here and not from the lane change state.
The model only reads the rising edge of a desire, so a pulse is one frame on, then off.
"""
import argparse
import numpy as np

from openpilot.cereal import log
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.tools.lib.framereader import FrameIterator
from openpilot.tools.lib.route import Route
from openpilot.tools.weight_lab.lab_paths import DATA
from openpilot.tools.weight_lab.replay import DT_MDL, LONG_SMOOTH_SECONDS, Runner, Warper, action_from_plan, latest, load_segment

DESIRES = {'none': log.Desire.none, 'left': log.Desire.turnLeft, 'right': log.Desire.turnRight}


def replay_with_desire(seg, runner, desire, start, every, stop):
  route_name, num = seg.rsplit('/', 1)
  rseg = Route(route_name).segments[int(num)]
  d = load_segment(seg)
  dc = DEVICE_CAMERAS[(d['device_type'], d['sensor'])]
  long_delay = d.get('long_delay', 0.0) + LONG_SMOOTH_SECONDS
  it_main = FrameIterator(rseg.camera_path, pix_fmt='nv12')
  it_wide = FrameIterator(rseg.ecamera_path, pix_fmt='nv12')
  W, H = dc.narrow_road.width, dc.narrow_road.height
  warp_main, warp_wide = Warper(W, H), Warper(W, H)
  cache = {'main': (-1, None), 'wide': (-1, None)}

  def frame_at(it, key, idx):
    cur, frm = cache[key]
    while cur < idx:
      frm = next(it)
      cur += 1
    cache[key] = (cur, frm)
    return frm

  rows = []
  prev_accel, prev_curv = 0.0, 0.0
  for mm in d['model']:
    fi_main = d['enc']['narrowRoadEncodeIdx'].get(mm['frameId'])
    fi_wide = d['enc']['wideRoadEncodeIdx'].get(mm.get('frameIdExtra', mm['frameId']))
    if fi_main is None or fi_wide is None:
      continue
    if fi_main < cache['main'][0] or fi_wide < cache['wide'][0]:
      continue
    if fi_main > stop:
      break
    t = mm['t']
    calib = latest(d['calib'], t, None)
    if calib is None:
      continue
    M_main = get_warp_matrix(calib, dc.narrow_road.intrinsics, False).astype(np.float32)
    M_wide = get_warp_matrix(calib, dc.wide_road.intrinsics, True).astype(np.float32)
    img = np.stack([warp_main(frame_at(it_main, 'main', fi_main), M_main),
                    warp_wide(frame_at(it_wide, 'wide', fi_wide), M_wide)]).astype(np.uint8)
    v_ego = max(latest(d['cs'], t, 0.0), 0.0)
    lat_t = latest(d['lat_delay'], t, 0.0) + DT_MDL + DT_MDL / 2
    long_t = long_delay + DT_MDL + DT_MDL / 2
    vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    # one frame on at start, then one frame on each `every` frames
    on = fi_main >= start and (fi_main == start or (every > 0 and (fi_main - start) % every == 0))
    vec_desire[DESIRES[desire] if on else 0] = 1
    parsed, raw = runner.step(img, vec_desire, d.get('rhd', False), [lat_t, long_t])
    plan = parsed['plan'][0]
    accel, curv = action_from_plan(plan, prev_accel, prev_curv, lat_t, long_t, v_ego)
    prev_accel, prev_curv = accel, curv
    rows.append(dict(t=t, frame=fi_main, v_ego=v_ego, accel=accel, curv=curv, plan=plan.astype(np.float32),
                     raw=raw.astype(np.float32), pulse=np.float32(on)) | runner.taps)
  return rows


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('seg')
  ap.add_argument('--desire', choices=list(DESIRES), required=True)
  ap.add_argument('--start', type=int, required=True)
  ap.add_argument('--every', type=int, default=0)
  ap.add_argument('--stop', type=int, default=10 ** 9)
  ap.add_argument('--onnx', default=f'{DATA}/tapped2.onnx')
  ap.add_argument('--out', required=True)
  args = ap.parse_args()
  rows = replay_with_desire(args.seg, Runner(args.onnx), args.desire, args.start, args.every, args.stop)
  print(args.seg, len(rows), 'frames', flush=True)
  np.savez_compressed(args.out, **{k: np.stack([np.asarray(r[k]) for r in rows]) for k in rows[0]})


if __name__ == '__main__':
  main()
