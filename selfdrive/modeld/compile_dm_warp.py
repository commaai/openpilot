#!/usr/bin/env python3
import argparse
import pickle
import time

from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.selfdrive.modeld.compile_modeld import NV12Frame, bind_camera_vars, make_camera_vars, warp_perspective_tinygrad, _parse_size


def make_warp_dm(dm_w, dm_h):
  def warp_dm(input_frame, M_inv, cam_w, cam_h, chroma_w, chroma_h, stride, uv_offset):
    M_inv = M_inv.to(Device.DEFAULT).realize()
    return warp_perspective_tinygrad(input_frame, M_inv, (dm_w, dm_h),
                                     (cam_h, cam_w), stride, border_fill_val=0).reshape(-1, dm_h * dm_w)
  return warp_dm


def compile_dm_warp(camera_configs: list[NV12Frame], dm_w, dm_h, pkl_path):
  print(f"Compiling DM warp for {len(camera_configs)} camera sizes -> {dm_w}x{dm_h}...")

  camera_vars, max_frame_size = make_camera_vars(camera_configs)
  warp_dm_jit = TinyJit(make_warp_dm(dm_w, dm_h), prune=True)

  for i in range(10):
    nv12 = camera_configs[i % len(camera_configs)]
    frame = Tensor.randint(max_frame_size, low=0, high=256, dtype='uint8').realize()
    M_inv = Tensor(Tensor.randn(3, 3).mul(8).realize().numpy(), device='NPY')
    Device.default.synchronize()
    st = time.perf_counter()
    warp_dm_jit(frame, M_inv, **bind_camera_vars(camera_vars, nv12)).realize()
    mt = time.perf_counter()
    Device.default.synchronize()
    et = time.perf_counter()
    print(f"  [{i+1}/10] {nv12.width}x{nv12.height} enqueue {(mt-st)*1e3:6.2f} ms -- total {(et-st)*1e3:6.2f} ms")

  with open(pkl_path, "wb") as f:
    pickle.dump({
      'warp': warp_dm_jit,
      'camera_configs': {nv12[:2]: nv12 for nv12 in camera_configs},
      'max_frame_size': max_frame_size,
    }, f)
  print(f"  Saved to {pkl_path}")


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument('--camera-resolution', type=_parse_size, help='camera resolution WxH')
  p.add_argument('--camera-resolutions', type=_parse_size, nargs='+', help='camera resolutions WxH (one or more)')
  p.add_argument('--warp-to', type=_parse_size, required=True, help='DM input WxH')
  p.add_argument('--output', required=True)
  args = p.parse_args()

  camera_resolutions = args.camera_resolutions or ([args.camera_resolution] if args.camera_resolution else None)
  assert camera_resolutions is not None, "one of --camera-resolution or --camera-resolutions is required"
  camera_configs = [NV12Frame(cam_w, cam_h, *get_nv12_info(cam_w, cam_h)) for cam_w, cam_h in camera_resolutions]
  dm_w, dm_h = args.warp_to
  compile_dm_warp(camera_configs, dm_w, dm_h, args.output)
