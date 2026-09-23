#!/usr/bin/env python3
import argparse
from contextlib import contextmanager
from dataclasses import replace
import gc
import json
import os
os.environ['GMMU'] = '0'
os.environ.setdefault('AM_POWER_LIMIT', '100')
from functools import partial
from pathlib import Path
import time

import numpy as np
from tinygrad import Context, Device, Tensor, TinyJit, dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.helpers import Target
from tinygrad.nn.onnx import OnnxRunner
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.uop.ops import AxisType, Ops
from tinygrad_repo.examples.openpilot.helpers import dump_pickle

from openpilot.selfdrive.modeld.worldmodel import WorldModel, load_weights
from openpilot.selfdrive.modeld.worldmodel_pkl import UPLOAD_CHUNK_SIZE


@contextmanager
def kernel_optimizations():
  import tinygrad.codegen as codegen

  config = json.loads(Path(__file__).with_name('worldmodel_kernel_opts.json').read_text())
  schedules = config['kernels'] if Device[Device.DEFAULT].arch == config['arch'] else {}
  original, applied = codegen.apply_opts, set()

  def apply_opts(ast, renderer, beam=0):
    key = ast.replace(arg=replace(ast.arg, beam=2)).key.hex()
    if ast.tag is None and (entry := schedules.get(key)) is not None:
      opts = []
      for op, axis, arg in entry['opts']:
        if isinstance(arg, list):
          arg = (arg[0], AxisType[arg[1]], *arg[2:]) if op == 'SPLIT' else tuple(arg)
        opts.append(Opt(OptOps[op], axis, arg))
      ast = ast.replace(arg=replace(ast.arg, opts_to_apply=tuple(opts)))
      applied.add(key)
    return original(ast, renderer, beam=beam)

  codegen.apply_opts = apply_opts
  try:
    yield
  finally:
    codegen.apply_opts = original
  print(f'Applied {len(applied)} tuned worldmodel kernel schedules', flush=True)


class WorldModelBuilder:
  def __init__(self, directory: Path, live_frames=5):
    hparams = json.loads((directory / 'hparams.json').read_text())
    context9 = hparams.get('openpilot_export', {}).get('format') == 'context9_fp8'
    if context9:
      live_frames = hparams['openpilot_export']['live_frames']
    assert 1 <= live_frames <= 10
    self.live_frames = live_frames
    device = Device[Device.DEFAULT]
    if device.arch not in ('gfx1200', 'gfx1201'):
      raise RuntimeError(f'Worldmodel requires an RDNA4 GPU, got {device.arch}')
    if device.is_usb:
      device.iface.dev_impl.smu.set_clocks(level=None)
    config = hparams['model']
    self.model = WorldModel(config, load_weights(directory / 'weights.fp8.safetensors', None, True))
    self.encoder = OnnxRunner(directory / 'encoder' / 'encoder.onnx')
    weights = {n.inputs[1] for n in self.encoder.graph_nodes if n.op == 'MatMul' and n.inputs[1] in self.encoder.const_names}
    for name in weights:
      self.encoder.graph_values[name] = self.encoder.graph_values[name].cast(dtypes.float16)
    self.encoder.onnx_ops = dict(self.encoder.onnx_ops)

    def matmul(a, b):
      b = b.cast(dtypes.float16)
      if not context9:
        b = b.contiguous().realize()
      return a.cast(dtypes.float16).contiguous().realize().matmul(
        b, dtype=dtypes.float32).realize()

    self.encoder.onnx_ops['MatMul'] = matmul
    for value in self.encoder.graph_values.values():
      if isinstance(value, Tensor):
        value.realize()
    sample = Tensor.zeros(1, 6, 128, 256).contiguous().realize()
    self.encoder({'imgs': sample})['latents'].realize()
    del sample

    self.prefix_frames = self.model.frames - live_frames
    fidx = (np.arange(self.model.frames, dtype=np.int64)[None] if context9 else
            np.array([[10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64))

    def conditions(start, end, timestep):
      return {
        't': Tensor.full((1, end - start), timestep, dtype=dtypes.bfloat16).contiguous().realize(),
        'augments_pos_ref_augment': Tensor.zeros(1, end - start, 3, dtype=dtypes.bfloat16).contiguous().realize(),
        'ref_augment_from_augments_euler': Tensor.zeros(1, end - start, 3, dtype=dtypes.bfloat16).contiguous().realize(),
        'pose_mask': Tensor.ones(1, end - start, dtype=dtypes.int64).contiguous().realize(),
        'fidx': Tensor(fidx[:, start:end].copy()).realize(),
      }

    # Unobserved future and older context slots have fixed noise at t=1.
    if self.prefix_frames:
      self.model.setup_cache(1, self.prefix_frames)
      rng = np.random.default_rng(0)
      prefix = Tensor(rng.standard_normal((1, self.prefix_frames, 32, 16, 32), dtype=np.float32)).cast(dtypes.bfloat16).realize()
      prefill = TinyJit(partial(self.model, **conditions(0, self.prefix_frames, 1), return_plan=False), prune=True)
      prefill.cnt = 1
      prefill(prefix)
      device.synchronize()
      del prefill, prefix
      gc.collect()
      device.allocator.free_cache()
    self.forward = partial(self.model, start_frame=self.prefix_frames, **conditions(self.prefix_frames, self.model.frames, 0))
    self.history = Tensor.zeros(1, live_frames, 32, 16, 32, dtype=dtypes.bfloat16).contiguous().realize()
    self.jit = TinyJit(self._run, prune=True)
    self.jit.cnt = 1
    self.run(np.zeros((1, 6, 128, 256), dtype=np.uint8), np.full((1, 2), .5, dtype=np.float32))
    self.reset_jit = TinyJit(self._reset)
    self.reset_jit.cnt = 1
    self.reset()

  def _run(self, images, action_t):
    images = images.to(Device.DEFAULT)
    action_t = action_t.to(Device.DEFAULT)
    latent = self.encoder({'imgs': images.float() / 127.5 - 1.0})['latents']
    latent = ((latent - self.model.config['compressor_mean']) / self.model.config['compressor_std']).cast(dtypes.bfloat16)
    self.history.assign(self.history[:, 1:].cat(latent.unsqueeze(1), dim=1)).realize()
    return {name: value.float().to('NPY').realize() for name, value in self.forward(self.history, action_t=action_t).items()}

  def _reset(self):
    self.history.assign(0).realize()

  def reset(self):
    self.reset_jit()

  def run(self, images, action_t):
    outputs = self.jit(Tensor(images, device='NPY'), Tensor(action_t, device='NPY'))
    outputs = {name: value.numpy() for name, value in outputs.items()}
    if not all(np.isfinite(value).all() for value in outputs.values()):
      raise RuntimeError('Worldmodel output is not finite')
    return outputs


def host_programs(jits, arch):
  renderer = ClangRenderer(Target('CPU', 'CLANG', arch))
  programs = {u for jit in jits.values() for u in jit.captured._linear.toposort()
              if u.op is Ops.PROGRAM and u.arg.target.device == 'CPU'}
  replacements = {p: to_program(p.replace(src=p.src[:2], arg=replace(p.arg, target=renderer.target)), renderer) for p in programs}
  return {name: type(jit)(None, replace(jit.captured, _linear=jit.captured._linear.substitute(replacements, enter_calls=True)))
          for name, jit in jits.items()}


def compile_model(directory: Path, output: Path):
  start = time.monotonic()
  with Context(PARALLEL=0), kernel_optimizations():
    runner = WorldModelBuilder(directory)
  inputs = np.random.default_rng(22).integers(0, 256, (32, 1, 6, 128, 256), dtype=np.uint8)
  action_t = np.random.default_rng(23).uniform(.3, .9, (32, 1, 2)).astype(np.float32)
  reference = []
  for index in range(64):
    reference.append({name: value.copy() for name, value in runner.run(inputs[index % 32], action_t[index % 32]).items()})
  runner.reset()

  @TinyJit
  def upload(destination, source):
    destination.assign(source.to(destination.device)).realize()

  source = Tensor(np.zeros(UPLOAD_CHUNK_SIZE, dtype=np.uint8), device='NPY')
  destination = source.to(Device.DEFAULT).realize()
  upload.cnt = 1
  upload(destination, source)
  Device[Device.DEFAULT].synchronize()
  jits = {'run': runner.jit, 'reset': runner.reset_jit, 'upload': upload}
  artifact = {
    'arch': Device[Device.DEFAULT].arch,
    'live_frames': runner.live_frames,
    'inputs': ('images', 'action_t'),
    'programs': {host: host_programs(jits, arch) for host, arch in
                 (('x86_64', 'x86_64,x86-64'), ('aarch64', 'arm64,generic'))},
    'metadata': json.loads((directory / 'hparams.json').read_text())['openpilot_export'],
  }
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = output.with_suffix(output.suffix + '.tmp')
  dump_pickle(artifact, temporary)
  temporary.replace(output)
  np.savez(output.with_suffix('.reference.npz'), inputs=inputs, action_t=action_t, allow_pickle=False,
           **{name: np.stack([r[name] for r in reference]) for name in reference[0]})
  print(f'Compiled {output}: {output.stat().st_size:,} bytes in {time.monotonic() - start:.2f} s', flush=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Precompile the FP8 worldmodel and USB transfers for ARM64 and x86-64 hosts.')
  parser.add_argument('directory', type=Path)
  parser.add_argument('output', type=Path)
  args = parser.parse_args()
  with Context(DEV='USB+AMD:LLVM', TC_OPT=2, TC_MIN_GLOBALS=32, JIT_BATCH_SIZE=0):
    compile_model(args.directory, args.output)
