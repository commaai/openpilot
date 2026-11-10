#!/usr/bin/env python3
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

os.environ.setdefault('DEV', 'USB+AMD:LLVM')
os.environ.setdefault('GMMU', '0')
os.environ.setdefault('AM_POWER_LIMIT', '100')

import numpy as np
import torch
from tinygrad import Device, Tensor, dtypes
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


def tensor(x, dtype=dtypes.bfloat16):
  return upload(x, dtype)


def sample_token(logits, config, rng, greedy):
  logits = logits.astype(np.float64).reshape(-1)
  logits[config.traj_ids['history_id0']:config.traj_ids['history_id0'] + config.history_vocab_size] = -np.inf
  logits[config.traj_ids['future_id0']:config.traj_ids['future_id0'] + config.future_vocab_size] = -np.inf
  logits[config.vlm_config.text_config.eos_token_id] = -np.inf
  assert np.isfinite(logits).any() and not np.isnan(logits).any()
  if greedy:
    return int(logits.argmax())
  logits /= 0.6
  order = np.argsort(-logits)
  probs = np.exp(logits[order] - np.max(logits))
  probs /= probs.sum()
  keep = np.cumsum(probs) - probs < 0.98
  return int(rng.choice(order[keep], p=probs[keep] / probs[keep].sum()))


def run_native(model, args, config, tokenizer, prepared, packed, seed):
  ids, positions, delta, vision_positions = packed
  times, generated = {}, []
  rng = np.random.default_rng(seed)
  start_total = time.perf_counter()
  with timed('input_upload', times, model.synchronize):
    pixels = tensor(prepared['tokenized_data']['pixel_values'].numpy())
    vpos = tensor(vision_positions)
  with timed('vision', times, model.synchronize):
    grids = prepared['tokenized_data']['image_grid_thw'].numpy()
    lengths = grids.prod(axis=1)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    image_features = []
    # Batching adjacent equal-sized images avoids padding or cross-image attention.
    i = 0
    while i < len(grids):
      end = i + 1
      while end < len(grids) and np.array_equal(grids[i], grids[end]) and end - i < args.vision_batch:
        end += 1
      shape = (end - i, int(lengths[i]), -1)
      image_features.append(model.vision(pixels[int(offsets[i]):int(offsets[end])].reshape(*shape),
                                          vpos[int(offsets[i]):int(offsets[end])].reshape(*shape)))
      i = end
    features = Tensor.cat(*image_features, dim=0)
    embedding = model.weights['vlm.model.language_model.embed_tokens.weight'][tensor(ids.astype(np.int32), dtypes.int32)]
    # Replace whole image spans so no dynamic boolean indexing runs on the GPU.
    chunks, cursor, feature_cursor = [], 0, 0
    image_indices = np.flatnonzero(ids[0] == config.vlm_config.image_token_id)
    for span in np.split(image_indices, np.flatnonzero(np.diff(image_indices) != 1) + 1):
      begin, end = int(span[0]), int(span[-1]) + 1
      chunks.extend((embedding[:, cursor:begin], features[feature_cursor:feature_cursor + end - begin].unsqueeze(0)))
      cursor, feature_cursor = end, feature_cursor + end - begin
    chunks.append(embedding[:, cursor:])
    embedding = Tensor.cat(*chunks, dim=1).realize()
  with timed('prefill', times, model.synchronize):
    cos, sin = (tensor(x) for x in rotary(positions))
    logits, cache = model.language(embedding, cos, sin)
  with timed('reasoning', times, model.synchronize):
    for step in range(args.max_tokens):
      token = sample_token(download(logits), config, rng, args.greedy)
      generated.append(token)
      pos = ids.shape[1] + step + delta
      cos, sin = (tensor(x) for x in rotary(np.full((3, 1, 1), pos, dtype=np.int64)))
      x = model.weights['vlm.model.language_model.embed_tokens.weight'][token].reshape(1, 1, -1).clone().realize()
      logits, cache = model.language(x, cos, sin, cache)
      if token == config.traj_ids['future_start']:
        break
    else:
      raise RuntimeError(f'No trajectory anchor after {args.max_tokens} tokens: {tokenizer.decode(generated)}')
  with timed('diffusion', times, model.synchronize):
    pos = np.arange(64) + ids.shape[1] + len(generated) + delta
    cos, sin = (tensor(x) for x in rotary(np.broadcast_to(pos, (3, 1, 64))))
    actions = tensor(rng.standard_normal((1, 64, 2)).astype(np.float32), dtypes.float32)
    flat_cache = [x.contiguous().realize() for pair in cache for x in pair]
    for step in range(args.diffusion_steps):
      velocity = model.expert_jit(actions, tensor([step / args.diffusion_steps], dtypes.float32), cos, sin, *flat_cache)
      actions = (actions + velocity / args.diffusion_steps).realize()
    action_array = download(actions)
  with timed('trajectory_decode', times, model.synchronize):
    action_config = dict(config.expert_config.action_space_cfg)
    action_config.pop('_target_')
    action_space = UnicycleAccelCurvatureCourseActionSpace(**action_config)
    with torch.no_grad():
      xyz, rot = action_space.action_to_traj(torch.from_numpy(action_array), prepared['ego_history_xyz'][:, 0], prepared['ego_history_rot'][:, 0])
    assert xyz.shape == (1, 64, 3) and rot.shape == (1, 64, 3, 3)
    assert torch.isfinite(xyz).all() and torch.isfinite(rot).all()
  times['total'] = (time.perf_counter() - start_total) * 1000
  return {'timings_ms': times, 'generated_tokens': len(generated), 'reasoning': tokenizer.decode(generated),
          'xyz': xyz.numpy().tolist(), 'rot': rot.numpy().tolist()}


def main():
  parser = argparse.ArgumentParser(description='Offline Alpamayo 2 Edge benchmark on Chestnut; does not publish controls.')
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
  parser.add_argument('--output', required=True, type=Path)
  args = parser.parse_args()
  assert args.runs > 0 and args.warmup >= 0 and args.diffusion_steps > 0 and args.vision_batch > 0
  torch.set_num_threads(8)
  data = synthetic_sample(args.profile) if args.synthetic else load_sample(args.input)
  started = time.perf_counter()
  config, tokenizer, prepared = prepare(args.checkpoint, data, args.profile)
  packed = native_inputs(args.checkpoint, config, prepared)
  preprocessing_ms = (time.perf_counter() - started) * 1000
  print(f'Input: {len(data["camera_indices"])} images, {packed[0].shape[1]} prompt tokens', flush=True)
  started = time.perf_counter()
  model = EdgeModel(args.checkpoint)
  model.synchronize()
  dev = Device[Device.DEFAULT]
  report = {'settings': {key: str(val) if isinstance(val, Path) else val for key, val in vars(args).items()},
            'environment': environment(), 'device': Device.DEFAULT, 'arch': getattr(dev, 'arch', None),
            'model_load_ms': (time.perf_counter() - started) * 1000, 'preprocessing_ms': preprocessing_ms,
            'prompt_tokens': packed[0].shape[1], 'warmup': [], 'runs': []}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  for i in range(args.warmup + args.runs):
    print(f'Iteration {i + 1}/{args.warmup + args.runs}', flush=True)
    result = run_native(model, args, config, tokenizer, prepared, packed, seed=0)
    report['warmup' if i < args.warmup else 'runs'].append(result)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
  totals = [run['timings_ms']['total'] for run in report['runs']]
  print(f'Median: {np.median(totals):.2f} ms, p95: {np.percentile(totals, 95):.2f} ms', flush=True)


if __name__ == '__main__':
  main()
