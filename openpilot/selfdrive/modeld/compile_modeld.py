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
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import round_up
from tinygrad.uop.ops import UOp


def nv12_copy_size(stride: int, y_height: int, uv_height: int) -> int:
  # Retain the padded Y and UV plane storage, but skip the trailing kernel/guard allocation.
  return stride * (y_height + uv_height)


def get_npy_shapes(input_shapes, state_pairs):
  shapes = {name: shape for name, (shape, _) in input_shapes.items() if name not in state_pairs and name != 'new_img'}
  return shapes, [math.prod(s) for s in shapes.values()]


def input_view(tensor: Tensor) -> Tensor:
  return Tensor(UOp.from_buffer(tensor._buffer())).reshape(tensor.shape)


def make_input_queues(input_shapes, state_pairs, device, frame_copy_size=0):
  shapes, sizes = get_npy_shapes(input_shapes, state_pairs)
  policy_size = sum(sizes) * np.dtype(np.float32).itemsize
  packed_input = np.zeros(round_up(128 + policy_size, 128) + 2 * frame_copy_size, dtype=np.uint8)
  packed_gpu = Tensor(packed_input, device=device).realize()
  policy = packed_input[128:128 + policy_size].view(np.float32)
  npy = {k: v.reshape(s) for (k, s), v in zip(shapes.items(), np.split(policy, np.cumsum(sizes[:-1])), strict=True)}
  input_queues = {name: Tensor(np.zeros(shape, dtype=dtype.fmt), device=device).realize()
                  for name, (shape, dtype) in input_shapes.items() if name in state_pairs or name == 'new_img'}
  input_queues['packed_npy_inputs'] = input_view(packed_gpu[128:128 + policy_size].bitcast('float32'))
  return input_queues, npy, packed_input, packed_gpu


def make_run_model(model_runner, input_shapes, state_pairs):
  shapes, sizes = get_npy_shapes(input_shapes, state_pairs)

  def run_model(new_img, packed_npy_inputs, **state_inputs):
    inputs = {name: t.reshape(s) for (name, s), t in zip(shapes.items(), packed_npy_inputs.split(sizes), strict=True)}
    inputs['new_img'] = new_img
    inputs = {name: value.cast(input_shapes[name][1]) for name, value in inputs.items()}
    outputs = {name: value.contiguous() for name, value in model_runner(inputs | state_inputs).items()}
    Tensor.realize(*outputs.values())
    if state_pairs:
      Tensor.realize(*(state_inputs[name].assign(outputs[next_name]) for name, next_name in state_pairs.items()))
    return tuple(value for name, value in outputs.items() if name not in state_pairs.values())
  return run_model


def compile_jit(jit, make_queues, benchmark_runs):
  if benchmark_runs < 1:
    raise ValueError("benchmark_runs must be at least 1")

  SEED = 42
  def random_inputs_run(fn, seed, n_runs, test_val=None, test_buffers=None, expect_match=True):
    input_queues, npy, packed_input, packed_gpu = make_queues(Device.DEFAULT)
    rng = np.random.default_rng(seed)

    for i in range(n_runs):
      for v in npy.values():
        v[:] = rng.standard_normal(v.shape).astype(v.dtype)
      packed_gpu._buffer().copy_from(Tensor(packed_input, device='NPY')._buffer())
      warped = rng.integers(0, 256, size=input_queues['new_img'].shape, dtype=np.uint8)
      input_queues['new_img'].assign(Tensor(warped, device=Device.DEFAULT)).realize()
      Device.default.synchronize()
      st = time.perf_counter()
      outs = fn(**input_queues)
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
  from openpilot.selfdrive.modeld.get_model_metadata import make_metadata_dict
  p = argparse.ArgumentParser()
  p.add_argument('--onnx', required=True)
  p.add_argument('--output', required=True)
  p.add_argument('--benchmark-runs', type=int, default=1,
                 help='timed loaded-JIT runs for each correctness seed')
  args = p.parse_args()

  model_path = read_file_chunked_to_disk(args.onnx)

  model_runner = OnnxRunner(model_path)
  input_shapes = {name: (spec.shape, spec.dtype) for name, spec in model_runner.graph_inputs.items()}
  state_pairs = {name: f'next_{name}' for name in input_shapes if f'next_{name}' in model_runner.graph_outputs}
  out = {
    'metadata': make_metadata_dict(model_path),
    'input_shapes': input_shapes,
    'state_pairs': state_pairs,
    'input_devices': {'model': Device.DEFAULT},
  }

  run_model = make_run_model(model_runner, input_shapes, state_pairs)
  make_model_queues = partial(make_input_queues, input_shapes, state_pairs)
  out['run_model'] = compile_jit(TinyJit(run_model, prune=True), make_model_queues, args.benchmark_runs)

  with open(args.output, "wb") as f:
    dump_oob(out, f)
  with open(args.output, "rb") as f:
    load_oob(f)
    assert not f.read(1), "unexpected model buffer data"
  print(f"Saved JITs to {args.output} ({os.path.getsize(args.output) / 1e6:.2f} MB)")
