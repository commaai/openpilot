#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from alpamayo2_edge.checkpoint import load_model
from alpamayo2_edge.helper import to_device
from inputs import load_sample, prepare, synthetic_sample
from report import environment


def main():
  parser = argparse.ArgumentParser(description='NVIDIA PyTorch reference benchmark, separate from the Chestnut runtime.')
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument('--profile', choices=('openpilot', 'nvidia'), default='openpilot')
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument('--input', type=Path)
  source.add_argument('--synthetic', action='store_true')
  parser.add_argument('--runs', type=int, default=3)
  parser.add_argument('--warmup', type=int, default=2)
  parser.add_argument('--greedy', action='store_true')
  parser.add_argument('--attention', choices=('eager', 'sdpa', 'flash_attention_2'), default='sdpa')
  parser.add_argument('--accel', choices=('off', 'compile', 'tuned'), default='off')
  parser.add_argument('--accel-attention', choices=('auto', 'sdpa', 'fa3'), default='auto')
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  assert args.runs > 0 and args.warmup >= 0
  torch.set_num_threads(8)
  model, _ = load_model(args.checkpoint, device=args.device, attn_implementation=args.attention)
  acceleration = None
  if args.accel != 'off':
    from alpamayo2_edge.accel import AccelConfig, enable_acceleration
    acceleration = enable_acceleration(model, AccelConfig(mode=args.accel, attention=args.accel_attention))
    assert not acceleration.skipped, acceleration.skipped
  data = synthetic_sample(args.profile) if args.synthetic else load_sample(args.input)
  _, tokenizer, prepared = prepare(args.checkpoint, data, args.profile)
  prepared = to_device(prepared, device=args.device)
  gpu = torch.cuda.get_device_properties(args.device)
  report = {'settings': {key: str(val) if isinstance(val, Path) else val for key, val in vars(args).items()},
            'environment': environment(), 'device': gpu.name, 'gpu_uuid': str(gpu.uuid),
            'prompt_tokens': prepared['tokenized_data']['input_ids'].shape[1], 'warmup': [], 'runs': []}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    for i in range(args.warmup + args.runs):
      print(f'Iteration {i + 1}/{args.warmup + args.runs}', flush=True)
      torch.manual_seed(0)
      torch.cuda.synchronize(args.device)
      started = time.perf_counter()
      xyz, rot, _, extra = model.sample_trajectories_from_data(prepared, return_extra=True, top_k=1 if args.greedy else None,
        diffusion_kwargs={'generator': torch.Generator(device=args.device).manual_seed(0)})
      torch.cuda.synchronize(args.device)
      elapsed = (time.perf_counter() - started) * 1000
      tokens = extra['generated_ids'][0]
      tokens = tokens[tokens >= 0]
      assert torch.isfinite(xyz).all() and torch.isfinite(rot).all()
      result = {'inference_ms': elapsed, 'generated_tokens': len(tokens), 'reasoning': tokenizer.decode(tokens),
                'xyz': xyz.cpu().float().tolist(), 'rot': rot.cpu().float().tolist()}
      report['warmup' if i < args.warmup else 'runs'].append(result)
      if acceleration is not None:
        report['acceleration'] = acceleration.describe()
      args.output.write_text(json.dumps(report, indent=2) + '\n')
      print(f'{i + 1}: {elapsed:.2f} ms, {len(tokens)} tokens', flush=True)
  totals = [r['inference_ms'] for r in report['runs']]
  print(f'Median: {np.median(totals):.2f} ms, p95: {np.percentile(totals, 95):.2f} ms')


if __name__ == '__main__':
  main()
