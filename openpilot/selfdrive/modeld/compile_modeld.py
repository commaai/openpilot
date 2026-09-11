#!/usr/bin/env python3
import argparse
import atexit
import math
import os
import tempfile
import time
import shutil
from functools import partial

import numpy as np

from openpilot.selfdrive.modeld.helpers import dump_oob, load_oob
from openpilot.selfdrive.modeld.compile_warp import NV12Frame, make_frame_prepare, parse_size as _parse_size


from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit


MODELD_INPUTS = ['img_q', 'big_img_q', 'feat_q', 'desire_q', 'packed_npy_inputs']


def nv12_copy_size(stride: int, y_height: int, uv_height: int) -> int:
  # Retain the padded Y and UV plane storage, but skip the trailing kernel/guard allocation.
  return stride * (y_height + uv_height)


def get_policy_npy_shapes(input_shapes):
  dp = input_shapes['desire_pulse']  # (1, 25, 8)
  tc = input_shapes['traffic_convention']  # (1, 2)
  at = input_shapes['action_t']  # (1, 2)
  fb = input_shapes['features_buffer']  # (1, T-1, ...) e.g. (1, 24, 32, 512) with spatial features
  feat_dim = math.prod(fb[2:])
  # TODO prev_feat shouldn't exist and be handled inside the JIT, but corrupt on QCOM for now
  shapes = {'desire': (dp[2],), 'traffic_convention': tuple(tc), 'action_t': tuple(at), 'prev_feat': (fb[0], feat_dim)}
  return shapes, [math.prod(s) for s in shapes.values()]


def make_input_queues(input_shapes, frame_skip, device, frame_copy_size):
  img = input_shapes['img']  # (1, 12, 128, 256)
  fb = input_shapes['features_buffer']  # (1, T-1, ...), past features only; the model appends the current frame's feature
  feat_dim = math.prod(fb[2:])
  dp = input_shapes['desire_pulse']  # (1, 25, 8)
  n_frames = img[1] // 6
  img_buf_shape = (frame_skip * (n_frames - 1) + 1, 6, img[2], img[3])

  policy_shapes, _ = get_policy_npy_shapes(input_shapes)
  shapes = {'tfm': (3, 3), 'big_tfm': (3, 3)} | policy_shapes
  sizes = [math.prod(s) for s in shapes.values()]
  packed_npy_size = sum(sizes) * np.dtype(np.float32).itemsize
  packed_input = np.zeros(packed_npy_size + 2 * frame_copy_size, dtype=np.uint8)
  packed_npy_inputs = packed_input[:packed_npy_size].view(np.float32)
  frames = packed_input[packed_npy_size:]
  frame_views = {'img': frames[:frame_copy_size], 'big_img': frames[frame_copy_size:]}
  # views into the packed inputs, to be refilled at runtime
  npy = {k: v.reshape(s) for (k, s), v in zip(shapes.items(), np.split(packed_npy_inputs, np.cumsum(sizes[:-1])), strict=True)}
  input_queues = {
    'img_q': Tensor(np.zeros(img_buf_shape, dtype=np.uint8), device=device).contiguous().realize(),
    'big_img_q': Tensor(np.zeros(img_buf_shape, dtype=np.uint8), device=device).contiguous().realize(),
    'feat_q': Tensor(np.zeros((frame_skip * fb[1], fb[0], feat_dim), dtype=np.float32), device=device).contiguous().realize(),
    'desire_q': Tensor(np.zeros((frame_skip * dp[1], dp[0], dp[2]), dtype=np.float32), device=device).contiguous().realize(),
    'packed_npy_inputs': Tensor(packed_input, device='NPY').realize(),
  }
  return input_queues, npy, frame_views


def shift_and_sample(buf, new_val, sample_fn):
  buf.assign(buf[1:].cat(new_val, dim=0).contiguous())
  return sample_fn(buf)


def sample_skip(buf, frame_skip):
  return buf[::frame_skip].contiguous().flatten(0, 1).unsqueeze(0)


def sample_desire(buf, frame_skip):
  return buf.reshape(-1, frame_skip, *buf.shape[1:]).max(1).flatten(0, 1).unsqueeze(0)


def make_warp(nv12, model_w, model_h):
  frame_prepare = make_frame_prepare(nv12, model_w, model_h)

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


def make_run_policy(model_runner, model_metadata, frame_skip):
  sample_desire_fn = partial(sample_desire, frame_skip=frame_skip)
  sample_skip_fn = partial(sample_skip, frame_skip=frame_skip)
  npy_shapes, npy_sizes = get_policy_npy_shapes(model_metadata['input_shapes'])
  model_input_dtypes = {name: spec.dtype for name, spec in model_runner.graph_inputs.items()}

  def run_policy(warped, img_q, big_img_q, feat_q, desire_q, packed_npy_inputs):
    packed_npy_inputs = packed_npy_inputs.to(Device.DEFAULT)
    Tensor.realize(packed_npy_inputs, warped)

    img = shift_and_sample(img_q, warped[0:1], sample_skip_fn)
    big_img = shift_and_sample(big_img_q, warped[1:2], sample_skip_fn)

    desire, traffic_convention, action_t, prev_feat = (t.reshape(s) for t, s in zip(packed_npy_inputs.split(npy_sizes), npy_shapes.values(), strict=True))
    desire_buf = shift_and_sample(desire_q, desire.reshape(1, 1, -1), sample_desire_fn)
    feat_buf = shift_and_sample(feat_q, prev_feat.reshape(1, 1, -1), sample_skip_fn)

    inputs = {
      'img': img,
      'big_img': big_img,
      'features_buffer': feat_buf.reshape(model_metadata['input_shapes']['features_buffer']),
      'desire_pulse': desire_buf,
      'traffic_convention': traffic_convention,
      'action_t': action_t,
    }
    inputs = {name: value.cast(model_input_dtypes[name]) for name, value in inputs.items()}
    out = next(iter(model_runner(inputs).values())).cast('float32')
    return out,
  return run_policy


def make_run_model(warp, run_policy, model_metadata, frame_copy_size):
  _, policy_sizes = get_policy_npy_shapes(model_metadata['input_shapes'])
  packed_npy_size = (18 + sum(policy_sizes)) * np.dtype(np.float32).itemsize

  def run_model(img_q, big_img_q, feat_q, desire_q, packed_npy_inputs):
    packed_input = packed_npy_inputs.to(Device.DEFAULT)
    Tensor.realize(packed_input)
    packed_npy_inputs = packed_input[:packed_npy_size].bitcast('float32')
    frame = packed_input[packed_npy_size:packed_npy_size + frame_copy_size]
    big_frame = packed_input[packed_npy_size + frame_copy_size:]
    tfm, big_tfm, policy_inputs = packed_npy_inputs.split([9, 9, sum(policy_sizes)])
    warped = warp(tfm.reshape(3, 3), big_tfm.reshape(3, 3), frame, big_frame)
    return run_policy(warped, img_q, big_img_q, feat_q, desire_q, policy_inputs)
  return run_model


def compile_jit(jit, input_keys, make_queues, benchmark_runs):
  if benchmark_runs < 1:
    raise ValueError("benchmark_runs must be at least 1")

  SEED = 42
  def random_inputs_run(fn, seed, n_runs, test_val=None, test_buffers=None, expect_match=True):
    input_queues, npy, frame_views = make_queues(Device.DEFAULT)
    rng = np.random.default_rng(seed)

    for i in range(n_runs):
      for v in npy.values():
        v[:] = rng.standard_normal(v.shape).astype(v.dtype)
      for v in frame_views.values():
        v[:] = rng.integers(0, 256, size=v.shape, dtype=np.uint8)
      Device.default.synchronize()
      st = time.perf_counter()
      outs = fn(**{k: input_queues[k] for k in input_keys})
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
  test_val, test_buffers = random_inputs_run(jit, SEED, 3)
  print(f'pickle round trip ({benchmark_runs} runs per seed)')
  with tempfile.TemporaryFile(dir=".") as f:
    dump_oob(jit, f)
    f.seek(0)
    loaded_jit = load_oob(f)
  random_inputs_run(loaded_jit, SEED, benchmark_runs, test_val, test_buffers, expect_match=True)
  random_inputs_run(loaded_jit, SEED+1, benchmark_runs, test_val, test_buffers, expect_match=False)
  # Keep the original so per-resolution JITs share model weight buffers in the final pickle.
  return jit


def read_file_chunked_to_disk(path):
  from openpilot.common.file_chunker import open_file_chunked
  tmp_path = f'{path}.unchunked'
  with open(tmp_path, 'wb') as f, open_file_chunked(path) as src:
    shutil.copyfileobj(src, f)
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
  p.add_argument('--benchmark-runs', type=int, default=1,
                 help='timed loaded-JIT runs for each correctness seed')
  args = p.parse_args()

  model_path = read_file_chunked_to_disk(args.onnx)
  model_w, model_h = args.model_size

  model_runner = OnnxRunner(model_path)
  out = {
    'metadata': make_metadata_dict(model_path),
    'input_devices': {'model': Device.DEFAULT},
    'run_model': {},
  }

  run_policy = make_run_policy(model_runner, out['metadata'], args.frame_skip)

  for cam_w, cam_h in args.camera_resolutions:
    nv12 = NV12Frame(cam_w, cam_h, *get_nv12_info(cam_w, cam_h))
    frame_copy_size = nv12_copy_size(nv12.stride, nv12.y_height, nv12.uv_height)
    make_model_queues = partial(make_input_queues, out['metadata']['input_shapes'], args.frame_skip,
                                frame_copy_size=frame_copy_size)
    warp = make_warp(nv12, model_w, model_h)
    run_model_jit = TinyJit(make_run_model(warp, run_policy, out['metadata'], frame_copy_size), prune=True)
    out['run_model'][(cam_w,cam_h)] = compile_jit(run_model_jit, MODELD_INPUTS, make_model_queues,
                                                  args.benchmark_runs)

  with open(args.output, "wb") as f:
    dump_oob(out, f)
  with open(args.output, "rb") as f:
    load_oob(f)
    assert not f.read(1), "unexpected model buffer data"
  print(f"Saved JITs to {args.output} ({os.path.getsize(args.output) / 1e6:.2f} MB)")
