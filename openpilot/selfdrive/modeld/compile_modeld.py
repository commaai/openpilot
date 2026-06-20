#!/usr/bin/env python3
import argparse
import atexit
import math
import os
import pickle
import tempfile
import time
from functools import partial
from collections import namedtuple

import numpy as np

def _patch_tinygrad_fetch_fw():
  import hashlib
  import pathlib
  import zstandard
  from tinygrad import helpers
  _orig = helpers.fetch_fw
  def fetch_fw(path, name, sha256):
    p = pathlib.Path(f"/lib/firmware/{path}/{name}.zst")
    if p.is_file():
      blob = zstandard.ZstdDecompressor().stream_reader(p.read_bytes()).read()
      if hashlib.sha256(blob).hexdigest() == sha256:
        return blob
    return _orig(path, name, sha256)
  helpers.fetch_fw = fetch_fw
_patch_tinygrad_fetch_fw()

from tinygrad.tensor import Tensor
from tinygrad.helpers import Context
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit


NV12Frame = namedtuple("NV12Frame", ['width', 'height', 'stride', 'y_height', 'uv_height', 'size'])
WARP_INPUTS = ['img_q', 'big_img_q', 'tfm', 'big_tfm']
POLICY_INPUTS = ['feat_q', 'desire_q', 'packed_npy_inputs']

UV_SCALE_MATRIX = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]], dtype=np.float32)
UV_SCALE_MATRIX_INV = np.linalg.inv(UV_SCALE_MATRIX)

WARP_DEV = os.getenv('WARP_DEV')


def make_random_images(keys, shape, device=None):
  return {k: Tensor.randint(shape, low=0, high=256, dtype='uint8', device=device).realize() for k in keys}


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
    # UV_SCALE @ M_inv @ UV_SCALE_INV simplifies to elementwise scaling
    M_inv_uv = M_inv * Tensor([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [2.0, 2.0, 1.0]], device=WARP_DEV)
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


def make_warp_input_queues(vision_input_shapes, frame_skip, device):
  img = vision_input_shapes['img']  # (1, 12, 128, 256)
  n_frames = img[1] // 6
  img_buf_shape = (frame_skip * (n_frames - 1) + 1, 6, img[2], img[3])

  npy = {
    'tfm': np.zeros((3, 3), dtype=np.float32),
    'big_tfm': np.zeros((3, 3), dtype=np.float32),
  }
  input_queues = {
    'img_q': Tensor(np.zeros(img_buf_shape, dtype=np.uint8), device=device).contiguous().realize(),
    'big_img_q': Tensor(np.zeros(img_buf_shape, dtype=np.uint8), device=device).contiguous().realize(),
    **{k: Tensor(v, device='NPY').realize() for k, v in npy.items()},
  }
  return input_queues, npy


def get_policy_npy_shapes(input_shapes):
  dp = input_shapes['desire_pulse']  # (1, 25, 8)
  tc = input_shapes['traffic_convention']  # (1, 2)
  at = input_shapes['action_t']  # (1, 2)
  fb = input_shapes['features_buffer']  # (1, 24, 512)
  # TODO prev_feat shouldn't exist and be handled inside the JIT, but corrupt on QCOM for now
  shapes = {'desire': (dp[2],), 'traffic_convention': tuple(tc), 'action_t': tuple(at), 'prev_feat': (fb[0], fb[2])}
  return shapes, [math.prod(s) for s in shapes.values()]


def make_input_queues(input_shapes, frame_skip, device):
  input_queues, npy = make_warp_input_queues(input_shapes, frame_skip, device)

  fb = input_shapes['features_buffer']  # (1, 24, 512), past features only; the model appends the current frame's feature
  dp = input_shapes['desire_pulse']  # (1, 25, 8)

  shapes, sizes = get_policy_npy_shapes(input_shapes)
  packed_npy_inputs = np.zeros(sum(sizes), dtype=np.float32)
  # views into the packed inputs, to be refilled at runtime
  npy.update({k: v.reshape(s) for (k, s), v in zip(shapes.items(), np.split(packed_npy_inputs, np.cumsum(sizes[:-1])), strict=True)})
  input_queues.update({
    'feat_q': Tensor(np.zeros((frame_skip * fb[1], fb[0], fb[2]), dtype=np.float32), device=device).contiguous().realize(),
    'desire_q': Tensor(np.zeros((frame_skip * dp[1], dp[0], dp[2]), dtype=np.float32), device=device).contiguous().realize(),
    'packed_npy_inputs': Tensor(packed_npy_inputs, device='NPY').realize(),
  })
  return input_queues, npy


def shift_and_sample(buf, new_val, sample_fn):
  buf.assign(buf[1:].cat(new_val, dim=0).contiguous())
  return sample_fn(buf)


def sample_skip(buf, frame_skip):
  return buf[::frame_skip].contiguous().flatten(0, 1).unsqueeze(0)


def sample_desire(buf, frame_skip):
  return buf.reshape(-1, frame_skip, *buf.shape[1:]).max(1).flatten(0, 1).unsqueeze(0)


def make_warp(nv12, model_w, model_h, frame_skip):
  frame_prepare = make_frame_prepare(nv12, model_w, model_h)
  sample_skip_fn = partial(sample_skip, frame_skip=frame_skip)

  def warp_enqueue(img_q, big_img_q, tfm, big_tfm, frame, big_frame):
    tfm = tfm.to(WARP_DEV)
    big_tfm = big_tfm.to(WARP_DEV)
    Tensor.realize(tfm, big_tfm)

    warped_frame = frame_prepare(frame, tfm).unsqueeze(0)
    warped_big_frame = frame_prepare(big_frame, big_tfm).unsqueeze(0)
    warped = Tensor.cat(warped_frame, warped_big_frame).to(Device.DEFAULT)
    img = shift_and_sample(img_q, warped[0:1], sample_skip_fn)
    big_img = shift_and_sample(big_img_q, warped[1:2], sample_skip_fn)
    return img, big_img
  return warp_enqueue


def make_run_policy(model_runner, model_metadata, frame_skip):
  sample_desire_fn = partial(sample_desire, frame_skip=frame_skip)
  sample_skip_fn = partial(sample_skip, frame_skip=frame_skip)
  npy_shapes, npy_sizes = get_policy_npy_shapes(model_metadata['input_shapes'])

  def run_policy(img, big_img, feat_q, desire_q, packed_npy_inputs):
    packed_npy_inputs = packed_npy_inputs.to(Device.DEFAULT).realize()
    desire, traffic_convention, action_t, prev_feat = (t.reshape(s) for t, s in zip(packed_npy_inputs.split(npy_sizes), npy_shapes.values(), strict=True))
    desire_buf = shift_and_sample(desire_q, desire.reshape(1, 1, -1), sample_desire_fn)
    feat_buf = shift_and_sample(feat_q, prev_feat.reshape(1, 1, -1), sample_skip_fn)

    inputs = {
      'img': img,
      'big_img': big_img,
      'features_buffer': feat_buf,
      'desire_pulse': desire_buf,
      'traffic_convention': traffic_convention,
      'action_t': action_t,
    }
    out = next(iter(model_runner(inputs).values())).cast('float32')
    return out,
  return run_policy


def compile_jit(jit, make_random_inputs, input_keys, make_queues):
  SEED = 42
  def random_inputs_run(fn, seed, test_val=None, test_buffers=None, expect_match=True):
    input_queues, npy = make_queues(Device.DEFAULT)
    np.random.seed(seed)
    Tensor.manual_seed(seed)

    testing = test_val is not None or test_buffers is not None
    n_runs = 1 if testing else 3

    for i in range(n_runs):
      for v in npy.values():
        v[:] = np.random.randn(*v.shape).astype(v.dtype)
      Device.default.synchronize()
      random_inputs = make_random_inputs()
      st = time.perf_counter()
      outs = fn(**{k: input_queues[k] for k in input_keys}, **random_inputs)
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
    return val, buffers

  print('capture + replay')
  test_val, test_buffers = random_inputs_run(jit, SEED)
  print('pickle round trip')
  jit = pickle.loads(pickle.dumps(jit))
  random_inputs_run(jit, SEED, test_val, test_buffers, expect_match=True)
  random_inputs_run(jit, SEED+1, test_val, test_buffers, expect_match=False)
  return jit


def _parse_size(s):
  w, h = s.lower().split('x')
  return int(w), int(h)


def read_file_chunked_to_shm(path):
  from openpilot.common.file_chunker import read_file_chunked
  from openpilot.common.hardware.hw import Paths
  with tempfile.NamedTemporaryFile(prefix='compile_modeld_', dir=Paths.shm_path(), delete=False) as f:
    f.write(read_file_chunked(path))
    tmp_path = f.name
  atexit.register(lambda: os.path.exists(tmp_path) and os.remove(tmp_path))
  return tmp_path


if __name__ == "__main__":
  from tinygrad.nn.onnx import OnnxRunner
  from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
  from openpilot.selfdrive.modeld.get_model_metadata import make_metadata_dict
  p = argparse.ArgumentParser()
  p.add_argument('--model-size', type=_parse_size, required=True, help='model input WxH')
  p.add_argument('--camera-resolutions', type=_parse_size, nargs='+', required=True,
                 help='camera resolutions WxH (one or more)')
  p.add_argument('--onnx', required=True)
  p.add_argument('--output', required=True)
  p.add_argument('--frame-skip', type=int, required=True)
  args = p.parse_args()

  model_path = read_file_chunked_to_shm(args.onnx)
  model_w, model_h = args.model_size

  model_runner = OnnxRunner(model_path)
  out = {'metadata': make_metadata_dict(model_path)}

  run_policy_jit = TinyJit(make_run_policy(model_runner, out['metadata'], args.frame_skip), prune=True)

  make_policy_queues = partial(make_input_queues, out['metadata']['input_shapes'], args.frame_skip)
  make_random_model_inputs = partial(make_random_images, keys=['img', 'big_img'], shape=out['metadata']['input_shapes']['img'])
  out['run_policy'] = compile_jit(run_policy_jit, make_random_model_inputs, POLICY_INPUTS,
                                  make_policy_queues)

  for cam_w, cam_h in args.camera_resolutions:
    nv12 = NV12Frame(cam_w, cam_h, *get_nv12_info(cam_w, cam_h))
    make_random_warp_inputs = partial(make_random_images, keys=['frame', 'big_frame'], shape=nv12.size, device=WARP_DEV)
    warp_enqueue = TinyJit(make_warp(nv12, model_w, model_h, args.frame_skip), prune=True)
    make_warp_queues = partial(make_warp_input_queues, out['metadata']['input_shapes'], args.frame_skip)
    out[(cam_w,cam_h)] = compile_jit(warp_enqueue, make_random_warp_inputs, WARP_INPUTS, make_warp_queues)

  with open(args.output, "wb") as f:
    pickle.dump(out, f)
  print(f"Saved JITs to {args.output} ({os.path.getsize(args.output) / 1e6:.2f} MB)")
