#!/usr/bin/env python3
import time
import pickle
from dataclasses import dataclass
from itertools import product
from functools import partial

import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import Context
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit

from openpilot.selfdrive.modeld.tinygrad_helpers import MODELS_DIR
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.common.transformations.model import MEDMODEL_INPUT_SIZE, DM_INPUT_SIZE
from openpilot.common.transformations.camera import _ar_ox_fisheye, _os_fisheye

# https://github.com/tinygrad/tinygrad/issues/15682
from tinygrad.uop.ops import UOp, Ops
_orig = UOp.__reduce__
UOp.__reduce__ = lambda self: (UOp.unique, ()) if self.op is Ops.UNIQUE else _orig(self)


@dataclass
class CompileConfig:
  cam_w: int
  cam_h: int
  prepare_only: bool
  prefix: str

  @property
  def pkl_path(self):
    return str(MODELS_DIR / f'{self.prefix}{"warp_" if self.prepare_only else ""}{self.cam_w}x{self.cam_h}_tinygrad.pkl')

CAMERA_CONFIGS = [
  (_ar_ox_fisheye.width, _ar_ox_fisheye.height),  # tici: 1928x1208
  (_os_fisheye.width, _os_fisheye.height),        # mici: 1344x760
]
MODELD_CONFIGS = [CompileConfig(cam_w, cam_h, prepare_only, 'driving_') for (cam_w, cam_h), prepare_only in product(CAMERA_CONFIGS, [True, False])]
DM_WARP_CONFIGS = [CompileConfig(cam_w, cam_h, True, 'dm_') for cam_w, cam_h in CAMERA_CONFIGS]


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


def make_frame_prepare(cam_w, cam_h, model_w, model_h):
  stride, y_height, uv_height, _ = get_nv12_info(cam_w, cam_h)
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


def make_warp_dm(cam_w, cam_h, dm_w, dm_h):
  stride, y_height, _, _ = get_nv12_info(cam_w, cam_h)
  stride_pad = stride - cam_w

  def warp_dm(input_frame, M_inv):
    M_inv = M_inv.to(Device.DEFAULT)
    result = warp_perspective_tinygrad(input_frame[:cam_h*stride], M_inv, (dm_w, dm_h), (cam_h, cam_w), stride_pad).reshape(-1, dm_h * dm_w)
    return result
  return warp_dm


def make_run_policy(vision_runner, policy_runner, cam_w, cam_h,
                    vision_features_slice, frame_skip, prepare_only=False):
  model_w, model_h = MEDMODEL_INPUT_SIZE
  frame_prepare = make_frame_prepare(cam_w, cam_h, model_w, model_h)
  sample_skip_fn = partial(sample_skip, frame_skip=frame_skip)
  sample_desire_fn = partial(sample_desire, frame_skip=frame_skip)

  def run_policy(img_q, big_img_q, feat_q, desire_q, desire, traffic_convention, tfm, big_tfm, frame, big_frame):
    img = shift_and_sample(img_q, frame_prepare(frame, tfm.to(Device.DEFAULT)).unsqueeze(0), sample_skip_fn)
    big_img = shift_and_sample(big_img_q, frame_prepare(big_frame, big_tfm.to(Device.DEFAULT)).unsqueeze(0), sample_skip_fn)

    if prepare_only:
      return img, big_img

    vision_out = next(iter(vision_runner({'img': img, 'big_img': big_img}).values())).cast('float32')

    new_feat = vision_out[:, vision_features_slice].reshape(1, -1).unsqueeze(0)
    feat_buf = shift_and_sample(feat_q, new_feat, sample_skip_fn)
    desire_buf = shift_and_sample(desire_q, desire.to(Device.DEFAULT).reshape(1, 1, -1), sample_desire_fn)

    inputs = {'features_buffer': feat_buf, 'desire_pulse': desire_buf, 'traffic_convention': traffic_convention.to(Device.DEFAULT)}
    policy_out = next(iter(policy_runner(inputs).values())).cast('float32')

    return vision_out, policy_out
  return run_policy


def compile_modeld(cam_w, cam_h, prepare_only, pkl_path):
  from tinygrad.nn.onnx import OnnxRunner
  from openpilot.selfdrive.modeld.constants import ModelConstants

  _, _, _, yuv_size = get_nv12_info(cam_w, cam_h)
  print(f"Compiling combined policy JIT for {cam_w}x{cam_h}...")

  vision_runner = OnnxRunner(MODELS_DIR / 'driving_vision.onnx')
  policy_runner = OnnxRunner(MODELS_DIR / 'driving_policy.onnx')

  with open(MODELS_DIR / 'driving_vision_metadata.pkl', 'rb') as f:
    vision_metadata = pickle.load(f)
    vision_features_slice = vision_metadata['output_slices']['hidden_state']
    vision_input_shapes = vision_metadata['input_shapes']
  with open(MODELS_DIR / 'driving_policy_metadata.pkl', 'rb') as f:
    policy_input_shapes = pickle.load(f)['input_shapes']

  frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ

  _run = make_run_policy(vision_runner, policy_runner,
                         cam_w, cam_h, vision_features_slice, frame_skip, prepare_only)
  run_policy_jit = TinyJit(_run, prune=True)

  N_RUNS = 3
  SEED = 42

  def random_inputs_run_fn(fn, seed, test_val=None, test_buffers=None, expect_match=True):
    input_queues, npy = make_input_queues(vision_input_shapes, policy_input_shapes, frame_skip)
    np.random.seed(seed)

    for i in range(N_RUNS):
      frame = Tensor(np.random.randint(0, 256, yuv_size, dtype=np.uint8)).realize()
      big_frame = Tensor(np.random.randint(0, 256, yuv_size, dtype=np.uint8)).realize()
      for v in npy.values():
        v[:] = np.random.randn(*v.shape).astype(v.dtype)
      Device.default.synchronize()
      st = time.perf_counter()
      outs = fn(**input_queues, frame=frame, big_frame=big_frame)
      mt = time.perf_counter()
      for o in outs:
        # .realize() not needed once jitted, but needed for unjitted fn
        o.realize()
      Device.default.synchronize()
      et = time.perf_counter()
      print(f"  [{i+1}/{N_RUNS}] enqueue {(mt-st)*1e3:6.2f} ms -- total {(et-st)*1e3:6.2f} ms")

    val = [np.copy(v.numpy()) for v in outs]
    buffers = [np.copy(v.numpy().copy()) for v in input_queues.values()]

    if test_val is not None:
      match = all(np.array_equal(a, b) for a, b in zip(val, test_val, strict=True))
      assert match == expect_match, f"outputs {'differ from' if expect_match else 'match'} baseline (seed={seed})"
    if test_buffers is not None:
      match = all(np.array_equal(a, b) for a, b in zip(buffers, test_buffers, strict=True))
      assert match == expect_match, f"buffers {'differ from' if expect_match else 'match'} baseline (seed={seed})"
    return fn, val, buffers

  print('run unjitted')
  _, test_val, test_buffers = random_inputs_run_fn(_run, seed=SEED)
  print('capture + replay')
  run_policy_jit, _, _ = random_inputs_run_fn(run_policy_jit, SEED, test_val, test_buffers)

  print('pickle round trip')
  with open(pkl_path, "wb") as f:
    pickle.dump(run_policy_jit, f)
    print(f"  Saved to {pkl_path}")
  with open(pkl_path, "rb") as f:
    run_policy_jit = pickle.load(f)
  random_inputs_run_fn(run_policy_jit, SEED, test_val, test_buffers, expect_match=True)
  random_inputs_run_fn(run_policy_jit, SEED+1, test_val, test_buffers, expect_match=False)


def compile_dm_warp(cam_w, cam_h, pkl_path):
  dm_w, dm_h = DM_INPUT_SIZE
  _, _, _, yuv_size = get_nv12_info(cam_w, cam_h)

  print(f"Compiling DM warp for {cam_w}x{cam_h}...")

  warp_dm = make_warp_dm(cam_w, cam_h, dm_w, dm_h)
  warp_dm_jit = TinyJit(warp_dm, prune=True)

  for i in range(10):
    inputs = [Tensor(np.random.randint(0, 256, yuv_size, dtype=np.uint8)).realize(),
              Tensor(Tensor.randn(3, 3).mul(8).realize().numpy(), device='NPY')]
    Device.default.synchronize()
    st = time.perf_counter()
    warp_dm_jit(*inputs)
    mt = time.perf_counter()
    Device.default.synchronize()
    et = time.perf_counter()
    print(f"  [{i+1}/10] enqueue {(mt-st)*1e3:6.2f} ms -- total {(et-st)*1e3:6.2f} ms")

  with open(pkl_path, "wb") as f:
    pickle.dump(warp_dm_jit, f)
  print(f"  Saved to {pkl_path}")


def run_and_save_pickle():
  for cfg in MODELD_CONFIGS:
    compile_modeld(cfg.cam_w, cfg.cam_h, cfg.prepare_only, cfg.pkl_path)
  for cfg in DM_WARP_CONFIGS:
    compile_dm_warp(cfg.cam_w, cfg.cam_h, cfg.pkl_path)


if __name__ == "__main__":
  run_and_save_pickle()
