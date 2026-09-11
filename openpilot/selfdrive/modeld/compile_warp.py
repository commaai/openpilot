import argparse
import pickle
import time
from collections import namedtuple
from tinygrad.tensor import Tensor
from tinygrad.helpers import Context
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info


NV12Frame = namedtuple("NV12Frame", ['width', 'height', 'stride', 'y_height', 'uv_height', 'size'])


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


def make_frame_prepare(nv12: NV12Frame, model_w, model_h, layout="yuv420", border_fill=None):
  cam_w, cam_h, stride, y_height, uv_height, _ = nv12
  uv_offset = stride * y_height
  stride_pad = stride - cam_w

  def frame_prepare_tinygrad(input_frame, M_inv):
    M_inv = M_inv.to(Device.DEFAULT).realize()
    if layout == "luma":
      return warp_perspective_tinygrad(input_frame[:cam_h*stride], M_inv,
                                      (model_w, model_h), (cam_h, cam_w), stride_pad,
                                      border_fill_val=border_fill).reshape(-1, model_h * model_w)
    # UV_SCALE @ M_inv @ UV_SCALE_INV simplifies to elementwise scaling
    M_inv_uv = M_inv * Tensor([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [2.0, 2.0, 1.0]], device=Device.DEFAULT)
    # deinterleave NV12 UV plane (UVUV... -> separate U, V)
    uv = input_frame[uv_offset:uv_offset + uv_height * stride].reshape(uv_height, stride)
    with Context(SPLIT_REDUCEOP=0):
      y = warp_perspective_tinygrad(input_frame[:cam_h*stride],
                                    M_inv, (model_w, model_h),
                                    (cam_h, cam_w), stride_pad, border_fill_val=border_fill).realize()
      u = warp_perspective_tinygrad(uv[:cam_h//2, :cam_w:2].flatten(),
                                    M_inv_uv, (model_w//2, model_h//2),
                                    (cam_h//2, cam_w//2), 0, border_fill_val=border_fill).realize()
      v = warp_perspective_tinygrad(uv[:cam_h//2, 1:cam_w:2].flatten(),
                                    M_inv_uv, (model_w//2, model_h//2),
                                    (cam_h//2, cam_w//2), 0, border_fill_val=border_fill).realize()
    yuv = y.cat(u).cat(v).reshape((model_h * 3 // 2, model_w))
    tensor = frames_to_tensor(yuv)
    return tensor
  return frame_prepare_tinygrad


def make_warp(nv12, model_w, model_h, layout="yuv420", border_fill=None):
  frame_prepare = make_frame_prepare(nv12, model_w, model_h, layout, border_fill)

  def warp(tfm, big_tfm, frame, big_frame):
    tfm = tfm.to(Device.DEFAULT)
    big_tfm = big_tfm.to(Device.DEFAULT)
    frame = frame.to(Device.DEFAULT)
    big_frame = big_frame.to(Device.DEFAULT)
    Tensor.realize(tfm, big_tfm, frame, big_frame)

    warped_frame = frame_prepare(frame, tfm).unsqueeze(0)
    warped_big_frame = frame_prepare(big_frame, big_tfm).unsqueeze(0)
    return Tensor.cat(warped_frame, warped_big_frame)

  return warp


def _parse_size(s):
  w, h = s.lower().split('x')
  return int(w), int(h)


def compile_warp(nv12: NV12Frame, model_w, model_h, pkl_path, layout, border_fill=None):
  print(f"Compiling {layout} warp for {nv12.width}x{nv12.height} -> {model_w}x{model_h}...")

  warp_jit = TinyJit(make_frame_prepare(nv12, model_w, model_h, layout, border_fill), prune=True)

  for i in range(10):
    frame = Tensor.randint(nv12.size, low=0, high=256, dtype='uint8').realize()
    M_inv = Tensor(Tensor.randn(3, 3).mul(8).realize().numpy(), device='NPY')
    Device.default.synchronize()
    st = time.perf_counter()
    warp_jit(frame, M_inv).realize()
    mt = time.perf_counter()
    Device.default.synchronize()
    et = time.perf_counter()
    print(f"  [{i+1}/10] enqueue {(mt-st)*1e3:6.2f} ms -- total {(et-st)*1e3:6.2f} ms")

  with open(pkl_path, "wb") as f:
    pickle.dump(warp_jit, f)
  print(f"  Saved to {pkl_path}")


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument('--camera-resolution', type=_parse_size, required=True, help='camera resolution WxH')
  p.add_argument('--warp-to', type=_parse_size, required=True, help='output WxH')
  p.add_argument('--layout', choices=['luma', 'yuv420'], required=True)
  p.add_argument('--border-fill', type=int, help='fill value outside the frame; omit to clamp coordinates')
  p.add_argument('--output', required=True)
  args = p.parse_args()

  cam_w, cam_h = args.camera_resolution
  nv12 = NV12Frame(cam_w, cam_h, *get_nv12_info(cam_w, cam_h))
  model_w, model_h = args.warp_to
  compile_warp(nv12, model_w, model_h, args.output, args.layout, args.border_fill)
