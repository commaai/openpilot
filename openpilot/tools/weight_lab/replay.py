#!/usr/bin/env python3
"""Replay the small driving model (ONNX) offline on a segment's full-res video.

Uses the frame ids, calibration, desire and delays that the device logged, so a replay of the
shipped model can be checked against the logged modelV2. Edits can then be scored on the same frames.

usage: replay.py ROUTE/SEG [ROUTE/SEG ...] --out out.npz [--onnx model.onnx]
"""
import argparse
import base64
import os
import pickle
import numpy as np
import onnxruntime as ort
from PIL import Image

from openpilot.cereal import log
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, get_curvature_from_plan, smooth_value
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.tools.lib.framereader import FrameIterator
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route
from openpilot.tools.weight_lab.lab_paths import DATA

DEFAULT_ONNX = '/home/batman/openpilot/openpilot/selfdrive/modeld/models/driving_supercombo.onnx'
DT_MDL = 0.05
LONG_SMOOTH_SECONDS = 0.3
MIN_LAT_CONTROL_SPEED = 0.3
MODEL_W, MODEL_H = 512, 256
UV_SCALE = np.array([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [2.0, 2.0, 1.0]], dtype=np.float32)
THUMB_DIR = f'{DATA}/viewer/thumbs'
THUMB_EVERY = 10  # frames, 0.5 s
LC = log.LaneChangeState
LCD = log.LaneChangeDirection


def warp_idx(M, w_dst, h_dst, w_src, h_src):
  """Nearest-pixel source index for each model pixel, same math as the tinygrad warp."""
  x = np.tile(np.arange(w_dst, dtype=np.float32), h_dst)
  y = np.repeat(np.arange(h_dst, dtype=np.float32), w_dst)
  sx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
  sy = M[1, 0] * x + M[1, 1] * y + M[1, 2]
  sw = M[2, 0] * x + M[2, 1] * y + M[2, 2]
  sx = np.clip(np.round(sx / sw), 0, w_src - 1).astype(np.int64)
  sy = np.clip(np.round(sy / sw), 0, h_src - 1).astype(np.int64)
  return sy * w_src + sx


class Warper:
  """Cache the gather indices per transform, the transform only changes with calibration."""
  def __init__(self, w, h):
    self.w, self.h = w, h
    self.key = None

  def __call__(self, nv12, M):
    key = M.tobytes()
    if key != self.key:
      self.key = key
      self.iy = warp_idx(M, MODEL_W, MODEL_H, self.w, self.h)
      self.iuv = warp_idx(M * UV_SCALE, MODEL_W // 2, MODEL_H // 2, self.w // 2, self.h // 2)
    w, h = self.w, self.h
    y_plane = nv12[:w * h]
    uv = nv12[w * h:w * h * 3 // 2].reshape(h // 2, w)
    y = y_plane[self.iy].reshape(MODEL_H, MODEL_W)
    u = uv[:, 0::2].reshape(-1)[self.iuv].reshape(MODEL_H // 2, MODEL_W // 2)
    v = uv[:, 1::2].reshape(-1)[self.iuv].reshape(MODEL_H // 2, MODEL_W // 2)
    return np.stack([y[0::2, 0::2], y[1::2, 0::2], y[0::2, 1::2], y[1::2, 1::2], u, v])


def save_thumb(nv12, w, h, path):
  """Quarter-res jpg from the NV12 frame, for the viewer."""
  y = nv12[:w * h].reshape(h, w)[::4, ::4]
  uv = nv12[w * h:w * h * 3 // 2].reshape(h // 2, w)
  u, v = uv[::2, 0::2][:, ::2], uv[::2, 1::2][:, ::2]
  ycc = np.stack([y, u[:y.shape[0], :y.shape[1]], v[:y.shape[0], :y.shape[1]]], axis=-1)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  Image.fromarray(ycc, 'YCbCr').convert('RGB').save(path, quality=70)


def desire_from_meta(meta):
  if meta.laneChangeState == LC.laneChangeStarting:
    if meta.laneChangeDirection == LCD.left:
      return log.Desire.laneChangeLeft
    if meta.laneChangeDirection == LCD.right:
      return log.Desire.laneChangeRight
  return log.Desire.none


def load_segment(seg):
  lr = LogReader(seg, sort_by_time=True)
  out = {'model': [], 'calib': [], 'lat_delay': [], 'cs': [], 'radar': []}
  enc = {'narrowRoadEncodeIdx': {}, 'wideRoadEncodeIdx': {}}
  for m in lr:
    w = m.which()
    if w == 'modelV2':
      out['model'].append(m.modelV2.as_builder().to_dict() | {'t': m.logMonoTime})
    elif w in enc:
      e = getattr(m, w)
      enc[w][e.frameId] = e.segmentId
    elif w == 'extrinsicsCalibration':
      out['calib'].append((m.logMonoTime, np.array(m.extrinsicsCalibration.rpyCalib, dtype=np.float32)))
    elif w == 'liveCalibration' and len(m.liveCalibration.rpyCalib) == 3:
      out['calib'].append((m.logMonoTime, np.array(m.liveCalibration.rpyCalib, dtype=np.float32)))
    elif w == 'lateralDelay':
      out['lat_delay'].append((m.logMonoTime, m.lateralDelay.lateralDelay))
    elif w == 'carState':
      out['cs'].append((m.logMonoTime, m.carState.vEgo))
    elif w == 'radarState':
      lead = m.radarState.leadOne
      ok = lead.present and lead.radar
      out['radar'].append((m.logMonoTime, np.array([ok, lead.dRel, lead.vLead, lead.aLeadK, lead.yRel] if ok else [0, np.nan, np.nan, np.nan, np.nan], dtype=np.float32)))
    elif w == 'carParams':
      out['long_delay'] = m.carParams.longitudinalActuatorDelay
    elif w == 'deviceState' and 'device_type' not in out:
      out['device_type'] = str(m.deviceState.deviceType)
    elif w == 'narrowRoadCameraState' and 'sensor' not in out:
      out['sensor'] = str(m.narrowRoadCameraState.sensor)
    elif w == 'driverMonitoringState':
      out['rhd'] = m.driverMonitoringState.isRHD
  out['enc'] = enc
  return out


def latest(series, t, default):
  val = default
  for ts, v in series:
    if ts > t:
      break
    val = v
  return val


class Runner:
  def __init__(self, onnx_path, provider='CUDAExecutionProvider'):
    ort.preload_dlls()
    self.sess = ort.InferenceSession(onnx_path, providers=[provider, 'CPUExecutionProvider'])
    meta = self.sess.get_modelmeta().custom_metadata_map
    self.slices = pickle.loads(base64.b64decode(meta['output_slices']))
    self.parser = Parser()
    self.reset()

  def reset(self):
    shapes = {i.name: i.shape for i in self.sess.get_inputs()}
    self.state = {k: np.zeros(shapes[k], dtype=np.uint8 if k == 'state_img_q' else np.float32)
                  for k in ('state_img_q', 'state_desire_q', 'state_feat_q')}
    self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)

  def step(self, new_img, desire_pulse, rhd, action_t):
    desire_pulse = desire_pulse.copy()
    desire_pulse[0] = 0
    desire = np.where(desire_pulse - self.prev_desire > .99, desire_pulse, 0).astype(np.float32)
    self.prev_desire = desire_pulse
    tc = np.zeros((1, 2), dtype=np.float32)
    tc[0, int(rhd)] = 1
    feeds = {'new_img': new_img, 'desire': desire, 'traffic_convention': tc,
             'action_t': np.array([action_t], dtype=np.float32)} | self.state
    outs = self.sess.run(None, feeds)
    names = [o.name for o in self.sess.get_outputs()]
    res = dict(zip(names, outs))
    for k in self.state:
      self.state[k] = res[f'next_{k}']
    raw = res['outputs'][0]
    parsed = self.parser.parse_outputs({k: raw[np.newaxis, v] for k, v in self.slices.items()})
    self.taps = {k: v.reshape(-1).astype(np.float32) for k, v in res.items() if k.startswith('x_')}
    return parsed, raw


def action_from_plan(plan, prev_accel, prev_curv, lat_t, long_t, v_ego):
  accel = get_accel_from_plan(plan[:, Plan.VELOCITY][:, 0], plan[:, Plan.ACCELERATION][:, 0], ModelConstants.T_IDXS, action_t=long_t)
  curv = get_curvature_from_plan(plan[:, Plan.T_FROM_CURRENT_EULER][:, 2], plan[:, Plan.ORIENTATION_RATE][:, 2],
                                 ModelConstants.T_IDXS, v_ego, lat_t)
  accel = smooth_value(accel, prev_accel, LONG_SMOOTH_SECONDS)
  if v_ego <= MIN_LAT_CONTROL_SPEED:
    curv = prev_curv
  return accel, curv


def replay_segment(seg, runner, keep_hidden=True):
  route_name, num = seg.rsplit('/', 1)
  rseg = Route(route_name).segments[int(num)]
  d = load_segment(seg)
  dc = DEVICE_CAMERAS[(d['device_type'], d['sensor'])]
  long_delay = d.get('long_delay', 0.0) + LONG_SMOOTH_SECONDS
  # decode both videos fully into memory as NV12 (1200 frames x 3.5 MB each would be too much, so stream)
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
  prev_desire = log.Desire.none
  for mi, mm in enumerate(d['model']):
    fi_main = d['enc']['narrowRoadEncodeIdx'].get(mm['frameId'])
    fi_wide = d['enc']['wideRoadEncodeIdx'].get(mm.get('frameIdExtra', mm['frameId']))
    if fi_main is None or fi_wide is None:
      continue
    if fi_main < cache['main'][0] or fi_wide < cache['wide'][0]:
      continue
    t = mm['t']
    calib = latest(d['calib'], t, None)
    if calib is None:
      continue
    M_main = get_warp_matrix(calib, dc.narrow_road.intrinsics, False).astype(np.float32)
    M_wide = get_warp_matrix(calib, dc.wide_road.intrinsics, True).astype(np.float32)
    main_frame = frame_at(it_main, 'main', fi_main)
    img = np.stack([warp_main(main_frame, M_main),
                    warp_wide(frame_at(it_wide, 'wide', fi_wide), M_wide)]).astype(np.uint8)
    if fi_main % THUMB_EVERY == 0:
      save_thumb(main_frame, W, H, f"{THUMB_DIR}/{route_name.replace('/', '_')}_{num}/{fi_main}.jpg")
    v_ego = max(latest(d['cs'], t, 0.0), 0.0)
    lat_t = latest(d['lat_delay'], t, 0.0) + DT_MDL + DT_MDL / 2
    long_t = long_delay + DT_MDL + DT_MDL / 2
    vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    vec_desire[prev_desire] = 1
    parsed, raw = runner.step(img, vec_desire, d.get('rhd', False), [lat_t, long_t])
    plan = parsed['plan'][0]
    accel, curv = action_from_plan(plan, prev_accel, prev_curv, lat_t, long_t, v_ego)
    prev_accel, prev_curv = accel, curv
    meta_obj = log.ModelDataV2.MetaData.new_message(**{k: mm['meta'][k] for k in ('laneChangeState', 'laneChangeDirection') if k in mm['meta']})
    prev_desire = desire_from_meta(meta_obj)
    rows.append(dict(
      t=t, frame=fi_main, v_ego=v_ego, big=bool(mm.get('big', False)),
      log_accel=mm['action']['desiredAcceleration'], log_curv=mm['action']['desiredCurvature'],
      log_plan_vx=np.array(mm['velocity']['x'], dtype=np.float32), log_plan_y=np.array(mm['position']['y'], dtype=np.float32),
      accel=accel, curv=curv, plan=plan.astype(np.float32), raw=raw.astype(np.float32),
      log_lead_v=np.float32(mm['leadsV3'][0]['v'][0] if mm.get('leadsV3') else np.nan),
      log_lead_x=np.float32(mm['leadsV3'][0]['x'][0] if mm.get('leadsV3') else np.nan),
      log_lead_prob=np.float32(mm['leadsV3'][0]['prob'] if mm.get('leadsV3') else np.nan),
      radar=latest(d['radar'], t, np.full(5, np.nan, dtype=np.float32)),
    ) | runner.taps)
  return rows


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('segs', nargs='+')
  ap.add_argument('--onnx', default=DEFAULT_ONNX)
  ap.add_argument('--out', required=True)
  args = ap.parse_args()
  runner = Runner(args.onnx)
  allrows = []
  for seg in args.segs:
    rows = replay_segment(seg, runner)
    for r in rows:
      r['seg'] = seg
    allrows += rows
    print(seg, len(rows), 'frames', flush=True)
  keys = [k for k in allrows[0] if k != 'seg']
  np.savez_compressed(args.out, seg=np.array([r['seg'] for r in allrows]), **{k: np.stack([np.asarray(r[k]) for r in allrows]) for k in keys})


if __name__ == '__main__':
  main()
