"""Compile perspective warps for NV12 frames."""
import argparse, pickle
from typing import NamedTuple
import numpy as np
from tinygrad import Tensor, Device, Context, TinyJit
from tinygrad.engine.realize import lower_and_compile
from examples.openpilot.helpers import allocate_inputs, benchmark, make_retargetable


class NV12Frame(NamedTuple):
  width: int
  height: int
  stride: int
  y_height: int
  uv_height: int
  size: int


def parse_frame(value): return NV12Frame(*map(int, value.split(',')))

def parse_size(value): return tuple(map(int, value.lower().split('x')))


def warp_perspective_tinygrad(src_flat, M_inv, dst_shape, src_shape, stride_pad, border_fill_val=None):
  w_dst, h_dst = dst_shape
  h_src, w_src = src_shape

  x = Tensor.arange(w_dst).reshape(1, w_dst).expand(h_dst, w_dst).reshape(-1)
  y = Tensor.arange(h_dst).reshape(h_dst, 1).expand(h_dst, w_dst).reshape(-1)

  # inline 3x3 matmul as elementwise to avoid reduce op (enables fusion with gather)
  src_x = M_inv[0, 0] * x + M_inv[0, 1] * y + M_inv[0, 2]
  src_y = M_inv[1, 0] * x + M_inv[1, 1] * y + M_inv[1, 2]
  src_w = M_inv[2, 0] * x + M_inv[2, 1] * y + M_inv[2, 2]

  src_x = src_x / src_w
  src_y = src_y / src_w

  x_round = Tensor.round(src_x)
  y_round = Tensor.round(src_y)
  x_nn_clipped = x_round.clip(0, w_src - 1).cast('int')
  y_nn_clipped = y_round.clip(0, h_src - 1).cast('int')
  idx = y_nn_clipped * (w_src + stride_pad) + x_nn_clipped
  sampled = src_flat[idx]

  if border_fill_val is None:
    return sampled

  in_bounds = ((x_round >= 0) & (x_round <= w_src - 1) &
               (y_round >= 0) & (y_round <= h_src - 1)).cast(sampled.dtype)
  return sampled * in_bounds + Tensor(border_fill_val, dtype=sampled.dtype) * (1 - in_bounds)


def frames_to_tensor(frames):
  H = (frames.shape[0] * 2) // 3
  W = frames.shape[1]
  in_img1 = Tensor.cat(frames[0:H:2, 0::2],
                       frames[1:H:2, 0::2],
                       frames[0:H:2, 1::2],
                       frames[1:H:2, 1::2],
                       frames[H:H+H//4].reshape((H//2, W//2)),
                       frames[H+H//4:H+H//2].reshape((H//2, W//2)), dim=0).reshape((6, H//2, W//2))
  return in_img1


def make_frame_prepare(nv12: NV12Frame, model_w, model_h):
  cam_w, cam_h, stride, y_height, uv_height, _ = nv12
  uv_offset = stride * y_height
  stride_pad = stride - cam_w

  def frame_prepare_tinygrad(input_frame, M_inv):
    M_inv = M_inv.to(Device.DEFAULT).realize()
    # UV_SCALE @ M_inv @ UV_SCALE_INV simplifies to elementwise scaling
    M_inv_uv = M_inv * Tensor([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [2.0, 2.0, 1.0]], device=Device.DEFAULT)
    # deinterleave NV12 UV plane (UVUV... -> separate U, V)
    uv = input_frame[uv_offset:uv_offset + uv_height * stride].reshape(uv_height, stride)
    with Context(SPLIT_REDUCEOP=0):
      y = warp_perspective_tinygrad(input_frame[:cam_h*stride],
                                    M_inv, (model_w, model_h),
                                    (cam_h, cam_w), stride_pad).realize()
      u = warp_perspective_tinygrad(uv[:cam_h//2, :cam_w:2].flatten(),
                                    M_inv_uv, (model_w//2, model_h//2),
                                    (cam_h//2, cam_w//2), 0).realize()
      v = warp_perspective_tinygrad(uv[:cam_h//2, 1:cam_w:2].flatten(),
                                    M_inv_uv, (model_w//2, model_h//2),
                                    (cam_h//2, cam_w//2), 0).realize()
    yuv = y.cat(u).cat(v).reshape((model_h * 3 // 2, model_w))
    tensor = frames_to_tensor(yuv)
    return tensor
  return frame_prepare_tinygrad


def make_luma_warp(nv12:NV12Frame, width, height, border_fill=None):
  def warp(input_frame, M_inv):
    M_inv = M_inv.to(Device.DEFAULT).realize()
    return warp_perspective_tinygrad(input_frame[:nv12.height*nv12.stride], M_inv,
                                    (width, height), (nv12.height, nv12.width), nv12.stride-nv12.width,
                                    border_fill_val=border_fill).reshape(-1, height*width)
  return warp

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--frame', type=parse_frame, required=True, help='width,height,stride,y_height,uv_height,buffer_size')
  parser.add_argument('--warp-to', type=parse_size, required=True)
  parser.add_argument('--layout', choices=['luma', 'yuv420'], default='luma')
  parser.add_argument('--border-fill', type=int, help='luma outside the frame; omit to clamp coordinates')
  parser.add_argument('--frames', type=int, default=1, help='number of frames to warp together')
  parser.add_argument('--transform-device', help='device holding the transforms; defaults to DEV')
  parser.add_argument('--output', required=True)
  parser.add_argument('--benchmark-runs', type=int, default=20)
  parser.add_argument('--retargetable', action='store_true', help='output retargetable jit (requires recompilation when loading)')
  args = parser.parse_args()

  warp = make_luma_warp(args.frame, *args.warp_to, args.border_fill) if args.layout == 'luma' else make_frame_prepare(args.frame, *args.warp_to)
  prefix = () if args.frames == 1 else (args.frames,)
  specs = {'input_frame': (prefix + (args.frame.size,), np.dtype(np.uint8).str, Device.DEFAULT),
           'M_inv': (prefix + (3, 3), np.dtype(np.float32).str, args.transform_device or Device.DEFAULT)}

  def make_inputs(seed):
    rng = np.random.default_rng(seed)
    def initialize(views):
      views['input_frame'][:] = rng.integers(0, 256, views['input_frame'].shape, dtype=np.uint8)
      views['M_inv'][:] = rng.standard_normal(views['M_inv'].shape)*8
    return allocate_inputs(specs, initialize)

  @TinyJit(prune=True)
  def run(input_frame, M_inv):
    if args.frames == 1: return warp(input_frame, M_inv)
    return Tensor.stack(*(warp(input_frame[i], M_inv[i]) for i in range(args.frames)))

  expected = benchmark(run, **(inputs:=make_inputs(42)))
  # capture jit
  for _ in range(2): np.testing.assert_array_equal(benchmark(run, **inputs), expected)
  # test jit output actually changes with different inputs
  with np.testing.assert_raises(AssertionError): np.testing.assert_array_equal(benchmark(run, **make_inputs(43)), expected)
  # benchmarks
  for i in range(args.benchmark_runs):
    np.testing.assert_array_equal(benchmark(run, cb=lambda t: print(f"  [{i}/{args.benchmark_runs}] {t*1e3:.2f} ms"), **inputs), expected)

  if args.retargetable: make_retargetable(run)

  artifact = {'metadata': {}, 'run': run, 'input_specs': specs}
  with open(args.output, 'wb') as f: pickle.dump(artifact, f)

  # test pickled jit
  with open(args.output, 'rb') as f: loaded = pickle.load(f)
  if args.retargetable: loaded['run'].captured._linear = lower_and_compile(loaded['run'].captured._linear)
  np.testing.assert_array_equal(benchmark(loaded['run'], **make_inputs(42)), expected)
