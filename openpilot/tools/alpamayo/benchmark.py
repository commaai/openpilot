#!/usr/bin/env python3
import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import time

os.environ.setdefault('DEV', 'USB+AMD:LLVM')
os.environ.setdefault('GMMU', '0')
os.environ.setdefault('AM_POWER_LIMIT', '100')
os.environ.setdefault('TC_OPT', '2')

import numpy as np
import torch
from tinygrad import Device, Tensor, TinyJit, dtypes
from alpamayo2_edge.action_space import UnicycleAccelCurvatureCourseActionSpace
from inputs import load_sample, native_inputs, prepare, rotary, synthetic_sample
from model import EdgeModel
from report import environment
from runtime import download, upload


@contextmanager
def timed(name, times, sync):
  sync()
  start = time.perf_counter()
  yield
  sync()
  times[name] = (time.perf_counter() - start) * 1000
  print(f'{name}: {times[name]:.2f} ms', flush=True)


def sample_token(logits, config, rng):
  logits = logits.astype(np.float64).reshape(-1)
  logits[config.traj_ids['history_id0']:config.traj_ids['history_id0'] + config.history_vocab_size] = -np.inf
  logits[config.traj_ids['future_id0']:config.traj_ids['future_id0'] + config.future_vocab_size] = -np.inf
  logits[config.vlm_config.text_config.eos_token_id] = -np.inf
  assert np.isfinite(logits).any() and not np.isnan(logits).any()
  logits /= 0.6
  order = np.argsort(-logits)
  probs = np.exp(logits[order] - np.max(logits))
  probs /= probs.sum()
  keep = np.cumsum(probs) - probs < 0.98
  return int(rng.choice(order[keep], p=probs[keep] / probs[keep].sum()))


class NativeInputs:
  def __init__(self, args, config, prepared, packed):
    ids, positions, delta, vision_positions = packed
    arrays = [prepared['tokenized_data']['pixel_values'].numpy(), vision_positions, ids.astype(np.int32), *rotary(positions)]
    for length in (args.max_tokens, args.max_tokens+64):
      pos = np.arange(length) + ids.shape[1] + delta
      arrays.extend(rotary(np.broadcast_to(pos, (3, 1, length))))
    arrays.append(np.zeros((1, 64, 2), dtype=np.float32))
    self.cpu = [Tensor(x, device='CPU').contiguous().realize() for x in arrays]
    self.noise = np.frombuffer(self.cpu[-1]._buffer().as_memoryview(allow_zero_copy=True), dtype=np.float32).reshape(1, 64, 2)
    types = [dtypes.bfloat16]*len(arrays)
    types[2], types[-1] = dtypes.int32, dtypes.float32
    self.upload_jit = TinyJit(lambda *xs: tuple(x.to(Device.DEFAULT).cast(dtype).contiguous().realize() for x, dtype in zip(xs, types, strict=True)))
    grids = prepared['tokenized_data']['image_grid_thw'].numpy()
    lengths = grids.prod(axis=1)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    batches, i = [], 0
    while i < len(grids):
      end = i+1
      while end < len(grids) and np.array_equal(grids[i], grids[end]) and end-i < args.vision_batch:
        end += 1
      batches.append((int(offsets[i]), int(offsets[end]), end-i))
      i = end
    self.batches = tuple(batches)
    image_indices = np.flatnonzero(ids[0] == config.vlm_config.image_token_id)
    self.spans = tuple((int(span[0]), int(span[-1])+1) for span in np.split(image_indices, np.flatnonzero(np.diff(image_indices) != 1)+1))

  def upload(self, noise):
    self.noise[:] = noise
    if int(os.getenv('ALPAMAYO_USB_PCIE', '0')):
      types = [dtypes.bfloat16]*len(self.cpu)
      types[2], types[-1] = dtypes.int32, dtypes.float32
      return tuple(upload(x, dtype) for x, dtype in zip(self.cpu, types, strict=True))
    return self.upload_jit(*self.cpu)


def run_native(model, args, config, tokenizer, prepared, packed, seed, inputs=None):
  if inputs is None:
    inputs = NativeInputs(args, config, prepared, packed)
  times, generated = {}, []
  rng = np.random.default_rng(seed)
  start_total = time.perf_counter()
  with timed('input_upload', times, model.synchronize):
    pixels, vpos, ids, cos, sin, decode_cos, decode_sin, action_cos, action_sin, actions = inputs.upload(rng.standard_normal((1, 64, 2)))
  with timed('vision', times, model.synchronize):
    embedding = model.encode(pixels, vpos, ids, inputs.batches, inputs.spans)
  with timed('prefill', times, model.synchronize):
    logits = model.prefill(embedding, cos, sin, args.max_tokens)
  with timed('reasoning', times, model.synchronize):
    for step in range(args.max_tokens):
      if args.greedy:
        logits, token_tensor = model.greedy_decode(logits, decode_cos, decode_sin, ids.shape[1] + step, step)
        token = int(download(token_tensor).item())
      else:
        token = sample_token(download(logits), config, rng)
        cos, sin = (x[:, :, step:step+1].clone().realize() for x in (decode_cos, decode_sin))
        x = model.weights['vlm.model.language_model.embed_tokens.weight'][token].reshape(1, 1, -1).clone().realize()
        logits = model.decode(x, cos, sin, ids.shape[1] + step)
      generated.append(token)
      if token == config.traj_ids['future_start']:
        break
    else:
      raise RuntimeError(f'No trajectory anchor after {args.max_tokens} tokens: {tokenizer.decode(generated)}')
  with timed('diffusion', times, model.synchronize):
    actions = model.trajectory(actions, action_cos, action_sin, ids.shape[1]+len(generated), args.diffusion_steps, len(generated))
    action_array = download(actions)
  with timed('trajectory_decode', times, model.synchronize):
    action_config = dict(config.expert_config.action_space_cfg)
    action_config.pop('_target_')
    action_space = UnicycleAccelCurvatureCourseActionSpace(**action_config)
    with torch.no_grad():
      xyz, rot = action_space.action_to_traj(torch.from_numpy(action_array.copy()), prepared['ego_history_xyz'][:, 0], prepared['ego_history_rot'][:, 0])
    assert xyz.shape == (1, 64, 3) and rot.shape == (1, 64, 3, 3)
    assert torch.isfinite(xyz).all() and torch.isfinite(rot).all()
  times['total'] = (time.perf_counter() - start_total) * 1000
  return {'timings_ms': times, 'generated_tokens': len(generated), 'reasoning': tokenizer.decode(generated),
          'xyz': xyz.numpy().tolist(), 'rot': rot.numpy().tolist()}


def main():
  parser = argparse.ArgumentParser(description='Offline Alpamayo 2 Edge benchmark in tinygrad; does not publish controls.')
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--profile', choices=('openpilot', 'nvidia'), default='openpilot')
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument('--input', type=Path)
  source.add_argument('--synthetic', action='store_true')
  parser.add_argument('--runs', type=int, default=3)
  parser.add_argument('--warmup', type=int, default=2)
  parser.add_argument('--max-tokens', type=int, default=256)
  parser.add_argument('--diffusion-steps', type=int, default=10)
  parser.add_argument('--vision-batch', type=int, default=2)
  parser.add_argument('--greedy', action='store_true')
  parser.add_argument('--precision', choices=('bf16', 'fp8'), default='bf16')
  parser.add_argument('--usb-pcie', action='store_true', help='Use slow synchronous PCIe transfers instead of tinygrad SRAM staging.')
  parser.add_argument('--output', required=True, type=Path)
  args = parser.parse_args()
  if args.usb_pcie:
    os.environ['ALPAMAYO_USB_PCIE'] = '1'
  assert args.runs > 0 and args.warmup >= 0 and args.diffusion_steps > 0 and args.vision_batch > 0
  torch.set_num_threads(8)
  data = synthetic_sample(args.profile) if args.synthetic else load_sample(args.input)
  started = time.perf_counter()
  config, tokenizer, prepared = prepare(args.checkpoint, data, args.profile)
  packed = native_inputs(args.checkpoint, config, prepared)
  inputs = NativeInputs(args, config, prepared, packed)
  preprocessing_ms = (time.perf_counter() - started) * 1000
  print(f'Input: {len(data["camera_indices"])} images, {packed[0].shape[1]} prompt tokens', flush=True)
  started = time.perf_counter()
  model = EdgeModel(args.checkpoint, precision=args.precision)
  model.synchronize()
  dev = Device[Device.DEFAULT]
  report = {'settings': {key: str(val) if isinstance(val, Path) else val for key, val in vars(args).items()},
            'environment': environment(), 'device': Device.DEFAULT, 'arch': getattr(dev, 'arch', None),
            'nvidia_gpu_index': getattr(getattr(dev, 'iface', None), 'gpu_minor', None),
            'fp8_linear_layers': len(model.weight_scales),
            'fp8_weights': sum(model.weights[name].numel() for name in model.weight_scales),
            'model_load_ms': (time.perf_counter() - started) * 1000, 'preprocessing_ms': preprocessing_ms,
            'prompt_tokens': packed[0].shape[1], 'warmup': [], 'runs': []}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  for i in range(args.warmup + args.runs):
    if i == args.warmup and args.warmup:
      started = time.perf_counter()
      gc.collect()
      report['warmup_cleanup_ms'] = (time.perf_counter() - started) * 1000
    print(f'Iteration {i + 1}/{args.warmup + args.runs}', flush=True)
    result = run_native(model, args, config, tokenizer, prepared, packed, seed=0, inputs=inputs)
    report['warmup' if i < args.warmup else 'runs'].append(result)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
  totals = [run['timings_ms']['total'] for run in report['runs']]
  print(f'Median: {np.median(totals):.2f} ms, p95: {np.percentile(totals, 95):.2f} ms', flush=True)


if __name__ == '__main__':
  main()
