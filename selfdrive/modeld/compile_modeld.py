#!/usr/bin/env python3
import argparse
import pickle
import time
from functools import partial
from collections import namedtuple

import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import Context
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.onnx import OnnxRunner

# https://github.com/tinygrad/tinygrad/issues/15682
from tinygrad.uop.ops import UOp, Ops
_orig = UOp.__reduce__
UOp.__reduce__ = lambda self: (UOp.unique, ()) if self.op is Ops.UNIQUE else _orig(self)


NV12Frame = namedtuple("NV12Frame", ['width', 'height', 'stride', 'y_height', 'uv_height', 'size'])

UV_SCALE_MATRIX = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]], dtype=np.float32)
UV_SCALE_MATRIX_INV = np.linalg.inv(UV_SCALE_MATRIX)


def warp_perspective_tinygrad(src_flat, M_inv, dst_shape, src_shape, stride_pad):
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

  x_nn_clipped = Tensor.round(src_x).clip(0, w_src - 1).cast('int')
  y_nn_clipped = Tensor.round(src_y).clip(0, h_src - 1).cast('int')
  idx = y_nn_clipped * (w_src + stride_pad) + x_nn_clipped

  return src_flat[idx]


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
    # UV_SCALE @ M_inv @ UV_SCALE_INV simplifies to elementwise scaling
    M_inv_uv = M_inv * Tensor([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [2.0, 2.0, 1.0]])
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


def make_input_queues(vision_input_shapes, policy_input_shapes, frame_skip):
  img = vision_input_shapes['img']  # (1, 12, 128, 256)
  n_frames = img[1] // 6
  img_buf_shape = (frame_skip * (n_frames - 1) + 1, 6, img[2], img[3])

  fb = policy_input_shapes['features_buffer']  # (1, 25, 512)
  dp = policy_input_shapes['desire_pulse']  # (1, 25, 8)
  tc = policy_input_shapes['traffic_convention']  # (1, 2)

  npy = {
    'desire': np.zeros(dp[2], dtype=np.float32),
    'traffic_convention': np.zeros(tc, dtype=np.float32),
    'tfm': np.zeros((3, 3), dtype=np.float32),
    'big_tfm': np.zeros((3, 3), dtype=np.float32),
  }
  input_queues = {
    'img_q': Tensor.zeros(img_buf_shape, dtype='uint8').contiguous().realize(),
    'big_img_q': Tensor.zeros(img_buf_shape, dtype='uint8').contiguous().realize(),
    'feat_q': Tensor.zeros(frame_skip * (fb[1] - 1) + 1, fb[0], fb[2]).contiguous().realize(),
    'desire_q': Tensor.zeros(frame_skip * dp[1], dp[0], dp[2]).contiguous().realize(),
    **{k: Tensor(v, device='NPY').realize() for k, v in npy.items()},
  }
  return input_queues, npy


def shift_and_sample(buf, new_val, sample_fn):
  buf.assign(buf[1:].cat(new_val, dim=0).contiguous())
  return sample_fn(buf)


def sample_skip(buf, frame_skip):
  return buf[::frame_skip].contiguous().flatten(0, 1).unsqueeze(0)


def sample_desire(buf, frame_skip):
  return buf.reshape(-1, frame_skip, *buf.shape[1:]).max(1).flatten(0, 1).unsqueeze(0)


def make_run_policy(vision_runner, policy_runner, nv12: NV12Frame, model_w, model_h,
                    vision_features_slice, frame_skip, prepare_only=False):
  frame_prepare = make_frame_prepare(nv12, model_w, model_h)
  sample_skip_fn = partial(sample_skip, frame_skip=frame_skip)
  sample_desire_fn = partial(sample_desire, frame_skip=frame_skip)

  def run_policy(img_q, big_img_q, feat_q, desire_q, desire, traffic_convention, tfm, big_tfm, frame, big_frame):
    tfm = tfm.to(Device.DEFAULT)
    big_tfm = big_tfm.to(Device.DEFAULT)
    desire = desire.to(Device.DEFAULT)
    traffic_convention = traffic_convention.to(Device.DEFAULT)
    Tensor.realize(tfm, big_tfm, desire, traffic_convention)

    img = shift_and_sample(img_q, frame_prepare(frame, tfm).unsqueeze(0), sample_skip_fn)
    big_img = shift_and_sample(big_img_q, frame_prepare(big_frame, big_tfm).unsqueeze(0), sample_skip_fn)

    if prepare_only:
      return img, big_img

    vision_out = next(iter(vision_runner({'img': img, 'big_img': big_img}).values())).cast('float32')

    new_feat = vision_out[:, vision_features_slice].reshape(1, -1).unsqueeze(0)
    feat_buf = shift_and_sample(feat_q, new_feat, sample_skip_fn)
    desire_buf = shift_and_sample(desire_q, desire.reshape(1, 1, -1), sample_desire_fn)

    inputs = {'features_buffer': feat_buf, 'desire_pulse': desire_buf, 'traffic_convention': traffic_convention}
    policy_out = next(iter(policy_runner(inputs).values())).cast('float32')

    return vision_out, policy_out
  return run_policy


def compile_modeld(nv12: NV12Frame, model_w, model_h, prepare_only, frame_skip,
                   vision_onnx, policy_onnx, pkl_path):
  from get_model_metadata import metadata_path_for

  print(f"Compiling combined policy JIT for {nv12.width}x{nv12.height} (prepare_only={prepare_only})...")

  vision_runner = OnnxRunner(vision_onnx)
  policy_runner = OnnxRunner(policy_onnx)

  with open(metadata_path_for(vision_onnx), 'rb') as f:
    vision_metadata = pickle.load(f)
    vision_features_slice = vision_metadata['output_slices']['hidden_state']
    vision_input_shapes = vision_metadata['input_shapes']
  with open(metadata_path_for(policy_onnx), 'rb') as f:
    policy_input_shapes = pickle.load(f)['input_shapes']

  _run = make_run_policy(vision_runner, policy_runner, nv12, model_w, model_h,
                         vision_features_slice, frame_skip, prepare_only)
  run_policy_jit = TinyJit(_run, prune=True)

  SEED = 42

  def random_inputs_run_fn(fn, seed, test_val=None, test_buffers=None, expect_match=True):
    input_queues, npy = make_input_queues(vision_input_shapes, policy_input_shapes, frame_skip)
    np.random.seed(seed)
    Tensor.manual_seed(seed)

    testing = test_val is not None or test_buffers is not None
    n_runs = 1 if testing else 3

    for i in range(n_runs):
      frame = Tensor.randint(nv12.size, low=0, high=256, dtype='uint8').realize()
      big_frame = Tensor.randint(nv12.size, low=0, high=256, dtype='uint8').realize()
      for v in npy.values():
        v[:] = np.random.randn(*v.shape).astype(v.dtype)
      Device.default.synchronize()
      st = time.perf_counter()
      outs = fn(**input_queues, frame=frame, big_frame=big_frame)
      mt = time.perf_counter()
      Device.default.synchronize()
      et = time.perf_counter()
      print(f"  [{i+1}/{n_runs}] enqueue {(mt-st)*1e3:6.2f} ms -- total {(et-st)*1e3:6.2f} ms")

      if i == 0:
        val = [np.copy(v.numpy()) for v in outs]
        buffers = [np.copy(v.numpy().copy()) for v in input_queues.values()]

    if test_val is not None:
      match = all(np.array_equal(a, b) for a, b in zip(val, test_val, strict=True))
      assert match == expect_match, f"outputs {'differ from' if expect_match else 'match'} baseline (seed={seed})"
    if test_buffers is not None:
      match = all(np.array_equal(a, b) for a, b in zip(buffers, test_buffers, strict=True))
      assert match == expect_match, f"buffers {'differ from' if expect_match else 'match'} baseline (seed={seed})"
    return fn, val, buffers

  print('capture + replay')
  run_policy_jit, test_val, test_buffers = random_inputs_run_fn(run_policy_jit, SEED)

  print('pickle round trip')
  with open(pkl_path, "wb") as f:
    pickle.dump(run_policy_jit, f)
    print(f"  Saved to {pkl_path}")
  with open(pkl_path, "rb") as f:
    run_policy_jit = pickle.load(f)
  random_inputs_run_fn(run_policy_jit, SEED, test_val, test_buffers, expect_match=True)
  random_inputs_run_fn(run_policy_jit, SEED+1, test_val, test_buffers, expect_match=False)


def _parse_size(s):
  w, h = s.lower().split('x')
  return int(w), int(h)


def _parse_nv12(s):
  parts = s.split(',')
  assert len(parts) == len(NV12Frame._fields), \
    f"--nv12 expects {','.join(NV12Frame._fields)} (got {s!r})"
  return NV12Frame(*(int(x) for x in parts))


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument('--model-size', type=_parse_size, required=True, help='model input WxH')
  p.add_argument('--nv12', type=_parse_nv12, required=True,
                 help=f'NV12 frame layout: {",".join(NV12Frame._fields)}')
  p.add_argument('--vision-onnx', required=True)
  p.add_argument('--policy-onnx', required=True)
  p.add_argument('--output', required=True)
  p.add_argument('--prepare-only', action='store_true')
  p.add_argument('--frame-skip', type=int, required=True)
  args = p.parse_args()

  model_w, model_h = args.model_size
  compile_modeld(args.nv12, model_w, model_h, args.prepare_only, args.frame_skip,
                 args.vision_onnx, args.policy_onnx, args.output)
