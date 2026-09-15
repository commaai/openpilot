#!/usr/bin/env python3
import argparse
import atexit
import os
import tempfile
import time
import shutil

import numpy as np

from openpilot.selfdrive.modeld.helpers import dump_oob, load_oob, patch_tinygrad_fetch_fw
patch_tinygrad_fetch_fw()

from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit


def make_input_queues(input_shapes, device):
  return {name: Tensor(np.zeros(shape, dtype=dtype.fmt), device=device).realize() for name, (shape, dtype) in input_shapes.items()}


def make_run_model(model_runner, state_pairs):
  def run_model(**inputs):
    outputs = {name: value.contiguous() for name, value in model_runner(inputs).items()}
    Tensor.realize(*outputs.values())
    if state_pairs:
      Tensor.realize(*(inputs[name].assign(outputs[next_name]) for name, next_name in state_pairs.items()))
    return tuple(value for name, value in outputs.items() if name not in state_pairs.values())
  return run_model


def compile_jit(jit, input_shapes, benchmark_runs):
  if benchmark_runs < 1:
    raise ValueError("benchmark_runs must be at least 1")

  SEED = 42
  def random_inputs_run(fn, seed, n_runs, test_val=None, test_buffers=None, expect_match=True):
    input_queues = make_input_queues(input_shapes, Device.DEFAULT)
    rng = np.random.default_rng(seed)

    for i in range(n_runs):
      for value in input_queues.values():
        values = rng.standard_normal(value.shape) if np.issubdtype(np.dtype(value.dtype.fmt), np.floating) else rng.integers(0, 256, value.shape)
        value.assign(Tensor(values.astype(value.dtype.fmt), device=Device.DEFAULT)).realize()
      Device.default.synchronize()
      st = time.perf_counter()
      outs = fn(**input_queues)
      mt = time.perf_counter()
      Device.default.synchronize()
      et = time.perf_counter()
      print(f"  [{i+1}/{n_runs}] enqueue {(mt-st)*1e3:6.2f} ms -- total {(et-st)*1e3:6.2f} ms")

      if i == 0:
        val = [v.numpy() for v in outs]
        buffers = [v.numpy() for v in input_queues.values()]

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

  run_model = make_run_model(model_runner, state_pairs)
  out['run_model'] = compile_jit(TinyJit(run_model, prune=True), input_shapes, args.benchmark_runs)

  with open(args.output, "wb") as f:
    dump_oob(out, f)
  with open(args.output, "rb") as f:
    load_oob(f)
    assert not f.read(1), "unexpected model buffer data"
  print(f"Saved JITs to {args.output} ({os.path.getsize(args.output) / 1e6:.2f} MB)")
