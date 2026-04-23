#!/usr/bin/env python3
import argparse
import pickle
import time

from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit

from openpilot.selfdrive.modeld.compile_modeld import NV12Frame, warp_perspective_tinygrad, _parse_size, _parse_nv12


def make_warp_dm(nv12: NV12Frame, dm_w, dm_h):
  cam_w, cam_h, stride, _, _, _ = nv12
  stride_pad = stride - cam_w

  def warp_dm(input_frame, M_inv):
    M_inv = M_inv.to(Device.DEFAULT).realize()
    return warp_perspective_tinygrad(input_frame[:cam_h*stride], M_inv,
                                     (dm_w, dm_h), (cam_h, cam_w), stride_pad).reshape(-1, dm_h * dm_w)
  return warp_dm


def compile_dm_warp(nv12: NV12Frame, dm_w, dm_h, pkl_path):
  print(f"Compiling DM warp for {nv12.width}x{nv12.height} -> {dm_w}x{dm_h}...")

  warp_dm_jit = TinyJit(make_warp_dm(nv12, dm_w, dm_h), prune=True)

  for i in range(10):
    frame = Tensor.randint(nv12.size, low=0, high=256, dtype='uint8').realize()
    M_inv = Tensor(Tensor.randn(3, 3).mul(8).realize().numpy(), device='NPY')
    Device.default.synchronize()
    st = time.perf_counter()
    warp_dm_jit(frame, M_inv).realize()
    mt = time.perf_counter()
    Device.default.synchronize()
    et = time.perf_counter()
    print(f"  [{i+1}/10] enqueue {(mt-st)*1e3:6.2f} ms -- total {(et-st)*1e3:6.2f} ms")

  with open(pkl_path, "wb") as f:
    pickle.dump(warp_dm_jit, f)
  print(f"  Saved to {pkl_path}")


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument('--nv12', type=_parse_nv12, required=True,
                 help=f'NV12 frame layout: {",".join(NV12Frame._fields)}')
  p.add_argument('--warp-to', type=_parse_size, required=True, help='DM input WxH')
  p.add_argument('--output', required=True)
  args = p.parse_args()

  dm_w, dm_h = args.warp_to
  compile_dm_warp(args.nv12, dm_w, dm_h, args.output)
