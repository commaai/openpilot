#!/usr/bin/env python3
"""
Benchmark modeld against real comma route data.

Downloads a route segment, feeds camera frames through ModelState,
and compares outputs against the actual modelV2 messages from the logs.
Catches frozen output bugs (from_blob/copyin), numerical drift, and regressions.

Usage:
  cd /data/openpilot
  PYTHONPATH=tinygrad_repo:. python3 selfdrive/modeld/benchmark_modeld.py
"""
import os
os.environ.setdefault('DEV', os.environ.get('DEV', 'CL'))

import sys, types

class _MockAttr:
  def __getattr__(self, name): return _MockAttr()
  def __call__(self, *a, **kw): return _MockAttr()
  def __getitem__(self, key): return _MockAttr()
  def __bool__(self): return False
  def __iter__(self): return iter([])

class _MockModule(types.ModuleType):
  def __init__(self, name):
    super().__init__(name)
    self.__path__, self.__file__ = [], f'<mock:{name}>'
  def __getattr__(self, name):
    if name.startswith('__') and name.endswith('__'): raise AttributeError(name)
    return _MockAttr()

for mock_name in [
    'msgq', 'msgq.ipc_pyx', 'msgq.visionipc',
    'cereal', 'cereal.messaging', 'cereal.car', 'cereal.log',
    'cereal.visionipc', 'zmq', 'serial',
    'opendbc', 'opendbc.car', 'opendbc.car.car_helpers',
    'openpilot.common.params_pyx', 'openpilot.common.params',
    'openpilot.common.swaglog',
]:
  if mock_name not in sys.modules:
    sys.modules[mock_name] = _MockModule(mock_name)

class _MockImporter:
  def find_module(self, name, path=None):
    if any(x in name for x in ['_pyx', 'params_pyx', 'swaglog', 'ipc_pyx']): return self
  def load_module(self, name):
    if name not in sys.modules: sys.modules[name] = _MockModule(name)
    return sys.modules[name]

sys.meta_path.insert(0, _MockImporter())

import argparse
import subprocess
import time
import json
import urllib.request
import numpy as np
from pathlib import Path

WARMUP_FRAMES = 20
MEASURE_FRAMES = 100
DEFAULT_ROUTE = "9748a98e983e0b39|0000002b--abc7a490ca"
DEFAULT_SEGMENT = 0


def align(val, a): return ((val + a - 1) // a) * a

def make_nv12_info(w, h):
  stride = align(w, 128)
  y_h, uv_h = align(h, 32), align(h // 2, 16)
  size = stride * y_h + stride * uv_h + 4096 + max(16 * 1024, 8 * stride)
  size = align(size, 4096) + align(w, 512) * 512
  return stride, y_h, uv_h, align(size, 4096)


class MockVisionBuf:
  def __init__(self, data, width, height):
    self.data = data
    self.width = width
    self.height = height


def decode_hevc(path, max_frames, width=None, height=None):
  if width is None or height is None:
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-show_entries', 'stream=width,height', '-of', 'csv=p=0:s=x', str(path)],
                       capture_output=True, timeout=30)
    width, height = [int(x) for x in r.stdout.decode().strip().split('x')]
  r = subprocess.run(['ffmpeg', '-nostdin', '-i', str(path), '-pix_fmt', 'nv12', '-f', 'rawvideo',
                       '-frames:v', str(max_frames), '-loglevel', 'error', 'pipe:1'],
                     capture_output=True, timeout=300)
  frame_size = width * height * 3 // 2
  raw = np.frombuffer(r.stdout, dtype=np.uint8)
  n = len(raw) // frame_size
  print(f"  Decoded {n} frames ({width}x{height}) from {Path(path).name}")
  return raw.reshape(n, frame_size), width, height


def pad_frame_nv12(raw_frame, w, h):
  stride, y_h, uv_h, yuv_size = make_nv12_info(w, h)
  padded = np.zeros(yuv_size, dtype=np.uint8)
  uv_offset = stride * y_h
  y_plane = raw_frame[:w * h].reshape(h, w)
  for row in range(h):
    padded[row * stride:row * stride + w] = y_plane[row]
  uv_plane = raw_frame[w * h:].reshape(h // 2, w)
  for row in range(h // 2):
    padded[uv_offset + row * stride:uv_offset + row * stride + w] = uv_plane[row]
  return padded


def download_route_segment(route_id, segment_num, output_dir):
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  needed = ['fcamera.hevc', 'ecamera.hevc', 'rlog.zst']
  if all((output_dir / f).exists() and (output_dir / f).stat().st_size > 1000 for f in needed):
    print(f"Route data already at {output_dir}")
    return output_dir
  route_encoded = route_id.replace('|', '%7C')
  api_url = f"https://api.comma.ai/v1/route/{route_encoded}/files"
  print(f"Fetching route files from {api_url}...")
  with urllib.request.urlopen(api_url, timeout=30) as resp:
    files = json.loads(resp.read())
  for filename, url in {'fcamera.hevc': files['cameras'][segment_num],
                         'ecamera.hevc': files['ecameras'][segment_num],
                         'rlog.zst': files['logs'][segment_num]}.items():
    dest = output_dir / filename
    if dest.exists() and dest.stat().st_size > 1000: continue
    print(f"  Downloading {filename}...")
    urllib.request.urlretrieve(url, str(dest))
  return output_dir


def parse_rlog(rlog_path):
  """Parse rlog.zst and extract calibration, camera info, and modelV2 reference outputs keyed by frameId."""
  import zstandard as zstd, capnp
  schema_path = Path(__file__).parents[2] / 'cereal' / 'log.capnp'
  log_schema = capnp.load(str(schema_path))

  with open(rlog_path, 'rb') as f:
    dat = f.read()
  if dat.startswith(b'\x28\xB5\x2F\xFD'):
    dat = zstd.ZstdDecompressor().stream_reader(dat).read()

  calib = [0.0, 0.0, 0.0]
  device_type, sensor = 'tici', 'ar0233'
  first_cam_frame_id = None
  reference_by_frame = {}

  for event in log_schema.Event.read_multiple_bytes(dat):
    try:
      w = event.which()
    except Exception:
      continue

    if w == 'liveCalibration':
      calib = list(event.liveCalibration.rpyCalib)
    elif w == 'deviceState':
      try: device_type = str(event.deviceState.deviceType)
      except Exception: pass
    elif w == 'roadCameraState':
      try: sensor = str(event.roadCameraState.sensor)
      except Exception: pass
      if first_cam_frame_id is None:
        try: first_cam_frame_id = event.roadCameraState.frameId
        except Exception: pass
    elif w == 'modelV2':
      mv2 = event.modelV2
      try:
        frame_id = mv2.frameId
        reference_by_frame[frame_id] = {
          'position_x': list(mv2.position.x),
          'position_y': list(mv2.position.y),
          'frameId': frame_id,
        }
      except Exception:
        pass

  print(f"  Parsed rlog: calib={[f'{c:.3f}' for c in calib]}, {len(reference_by_frame)} modelV2 msgs, "
        f"first_cam_frameId={first_cam_frame_id}")
  return calib, device_type, sensor, first_cam_frame_id or 0, reference_by_frame


def main():
  parser = argparse.ArgumentParser(description='Benchmark modeld against route data')
  parser.add_argument('--segment', help='Path to local segment dir (fcamera.hevc + ecamera.hevc + rlog.zst)')
  parser.add_argument('--route', default=DEFAULT_ROUTE, help='Comma connect route ID to download')
  parser.add_argument('--route-segment', type=int, default=DEFAULT_SEGMENT)
  parser.add_argument('--data-dir', default='/home/radxa/segment')
  parser.add_argument('--warmup', type=int, default=WARMUP_FRAMES)
  parser.add_argument('--frames', type=int, default=MEASURE_FRAMES)
  args = parser.parse_args()

  n_total = args.warmup + args.frames

  # Download or locate route segment
  seg = Path(args.segment) if args.segment else download_route_segment(args.route, args.route_segment, args.data_dir)

  # Decode video frames
  raw_road, cam_w, cam_h = decode_hevc(seg / 'fcamera.hevc', n_total)
  raw_wide, _, _ = decode_hevc(seg / 'ecamera.hevc', n_total, cam_w, cam_h)
  n_total = min(n_total, len(raw_road), len(raw_wide))
  road_frames = [pad_frame_nv12(raw_road[i], cam_w, cam_h) for i in range(n_total)]
  wide_frames = [pad_frame_nv12(raw_wide[i], cam_w, cam_h) for i in range(n_total)]

  # Parse reference outputs from rlog
  rlog = seg / 'rlog.zst'
  if not rlog.exists():
    print(f"ERROR: {rlog} not found — need rlog for reference comparison")
    sys.exit(1)
  calib, device_type, sensor, first_cam_frame_id, reference_by_frame = parse_rlog(rlog)
  if len(reference_by_frame) < 10:
    print(f"ERROR: only {len(reference_by_frame)} modelV2 msgs in rlog, need at least 10")
    sys.exit(1)

  # Compute transforms from log calibration
  from openpilot.common.transformations.model import get_warp_matrix
  from openpilot.common.transformations.camera import DEVICE_CAMERAS
  dc_key = (device_type, sensor)
  if dc_key not in DEVICE_CAMERAS:
    dc_key = next(iter(DEVICE_CAMERAS))
    print(f"  Warning: camera config {(device_type, sensor)} not found, using {dc_key}")
  dc = DEVICE_CAMERAS[dc_key]
  calib_np = np.array(calib, dtype=np.float32)
  transform_main = get_warp_matrix(calib_np, dc.fcam.intrinsics, False).astype(np.float32)
  transform_extra = get_warp_matrix(calib_np, dc.ecam.intrinsics, True).astype(np.float32)
  print(f"Camera: {cam_w}x{cam_h}, config={dc_key}, calib={[f'{c:.3f}' for c in calib]}")

  # Load model
  print("Loading ModelState...")
  t0 = time.monotonic()
  from selfdrive.modeld.modeld import ModelState
  model = ModelState(cam_w, cam_h)
  print(f"Model loaded in {time.monotonic() - t0:.1f}s")

  input_names = model.vision_input_names
  measured_frames = min(args.frames, n_total - args.warmup)
  print(f"Running {n_total} frames ({args.warmup} warmup + {measured_frames} measured)...")

  times = []
  our_outputs = []

  for i in range(n_total):
    idx = i % len(road_frames)
    frame_id = first_cam_frame_id + i
    road_buf = MockVisionBuf(road_frames[idx], cam_w, cam_h)
    wide_buf = MockVisionBuf(wide_frames[idx], cam_w, cam_h)
    bufs = {name: wide_buf if 'big' in name else road_buf for name in input_names}
    transforms = {name: transform_extra if 'big' in name else transform_main for name in input_names}
    inputs = {'desire_pulse': np.zeros(8, dtype=np.float32), 'traffic_convention': np.array([1, 0], dtype=np.float64)}

    t1 = time.perf_counter()
    result = model.run(bufs, transforms, inputs, prepare_only=False)
    elapsed_ms = (time.perf_counter() - t1) * 1000

    if i >= args.warmup and result is not None:
      times.append(elapsed_ms)
      our_outputs.append({
        'position_x': result['plan'][0, :, 0].copy(),
        'position_y': result['plan'][0, :, 1].copy(),
        'hidden_state': result['hidden_state'].copy(),
        'frame_id': frame_id,
      })

    if (i + 1) % 100 == 0:
      med = np.median(times) if times else 0
      print(f"  {i + 1}/{n_total} | median: {med:.1f}ms")

  if len(times) < 5:
    print("ERROR: too few measured frames")
    print("RESULT median_ms=0.0 status=fail reason=too_few_frames")
    sys.exit(1)

  # === CHECKS ===
  times_arr = np.array(times)
  median_ms = float(np.median(times_arr))
  p95_ms = float(np.percentile(times_arr, 95))
  min_ms = float(np.min(times_arr))
  failures = []

  # 1. Frozen output detection
  if len(our_outputs) >= 5:
    for key in ['position_x', 'hidden_state']:
      stacked = np.array([o[key].flatten() for o in our_outputs[-10:]])
      frame_std = np.std(stacked, axis=0).mean()
      if frame_std < 1e-6:
        failures.append(f"FROZEN: {key} identical across frames (std={frame_std:.2e}) — from_blob/copyin bug?")
        print(f"  FAIL: {key} FROZEN (std={frame_std:.2e})")
      else:
        print(f"  OK: {key} varies across frames (std={frame_std:.4f})")

  # 2. Reference comparison against log's modelV2 (aligned by frameId)
  cosines = []
  for ours in our_outputs:
    fid = ours['frame_id']
    if fid not in reference_by_frame:
      continue
    ref = reference_by_frame[fid]
    ref_x = np.array(ref['position_x'], dtype=np.float64)
    our_x = ours['position_x'].flatten()[:len(ref_x)].astype(np.float64)
    dot = np.dot(ref_x, our_x)
    norm = np.linalg.norm(ref_x) * np.linalg.norm(our_x)
    cosines.append(float(dot / norm) if norm > 1e-10 else 0.0)
  if len(cosines) >= 5:
    cosine_avg = float(np.mean(cosines))
    cosine_min = float(np.min(cosines))
    print(f"  Reference match: cosine avg={cosine_avg:.4f} min={cosine_min:.4f} ({len(cosines)} frames matched by frameId)")
    if cosine_avg < 0.90:
      failures.append(f"position_x cosine vs log too low: avg={cosine_avg:.4f}")
  else:
    cosine_avg = None
    print(f"  Warning: only {len(cosines)} frames matched by frameId for reference check")

  # 3. Output sanity
  if our_outputs:
    sample = our_outputs[len(our_outputs) // 2]
    for key in ['position_x', 'position_y']:
      arr = sample[key]
      if np.all(arr == 0) or np.any(np.isnan(arr)):
        failures.append(f"{key} is all zeros or NaN")
      else:
        print(f"  {key}: range=[{arr.min():.2f}, {arr.max():.2f}]")

  status = "fail" if failures else "pass"

  print(f"\n{'='*60}")
  print(f"Timing ({len(times)} frames): median={median_ms:.1f}ms  p95={p95_ms:.1f}ms  min={min_ms:.1f}ms")
  if cosine_avg is not None:
    print(f"Reference accuracy: cosine avg={cosine_avg:.4f} min={cosine_min:.4f}")
  if failures:
    print(f"FAILURES:")
    for f in failures:
      print(f"  - {f}")
  else:
    print("All checks passed")
  print(f"{'='*60}")
  cosine_str = f"cosine={cosine_avg:.4f}" if cosine_avg is not None else "cosine=n/a"
  cosine_min_str = f"cosine_min={cosine_min:.4f}" if cosine_avg is not None else ""
  reason = failures[0] if failures else "all_checks_passed"
  print(f"RESULT median_ms={median_ms:.1f} p95_ms={p95_ms:.1f} {cosine_str} {cosine_min_str} status={status} reason={reason}")

  sys.exit(0 if status == "pass" else 1)


if __name__ == '__main__':
  main()
