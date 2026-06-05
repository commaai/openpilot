#!/usr/bin/env python3
import argparse
from tqdm import tqdm

from cereal.services import SERVICE_LIST, QueueSize
from openpilot.tools.lib.logreader import LogReader


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Analyze message sizes from a log route")
  parser.add_argument("route", nargs="?", default="98395b7c5b27882e/000000a8--f87e7cd255",
                      help="Log route to analyze (default: 98395b7c5b27882e/000000a8--f87e7cd255)")
  args = parser.parse_args()

  lr = LogReader(args.route)

  szs = {}
  for msg in tqdm(lr):
    sz = len(msg.as_builder().to_bytes())
    msg_type = msg.which()
    if msg_type not in szs:
      szs[msg_type] = {'min': sz, 'max': sz, 'sum': sz, 'count': 1}
    else:
      szs[msg_type]['min'] = min(szs[msg_type]['min'], sz)
      szs[msg_type]['max'] = max(szs[msg_type]['max'], sz)
      szs[msg_type]['sum'] += sz
      szs[msg_type]['count'] += 1

  print()
  print(f"{'Service':<36} {'Min (KB)':>12} {'Max (KB)':>12} {'Avg (KB)':>12} {'KB/min':>12} {'KB/sec':>12} {'Minutes in 10MB':>18} {'Seconds in Queue':>18}")
  print("-" * 132)
  def sort_key(x):
    k, v = x
    avg = v['sum'] / v['count']
    freq = SERVICE_LIST.get(k, None)
    freq_val = freq.frequency if freq else 0.0
    kb_per_min = (avg * freq_val * 60) / 1024 if freq_val > 0 else 0.0
    return kb_per_min
  total_kb_per_min = 0.0
  RINGBUFFER_SIZE_KB = 10 * 1024  # 10MB old default
  for k, v in sorted(szs.items(), key=sort_key, reverse=True):
    avg = v['sum'] / v['count']
    service = SERVICE_LIST.get(k, None)
    freq_val = service.frequency if service else 0.0
    queue_size_kb = (service.queue_size / 1024) if service else 250  # default to SMALL
    kb_per_min = (avg * freq_val * 60) / 1024 if freq_val > 0 else 0.0
    kb_per_sec = kb_per_min / 60
    minutes_in_buffer = RINGBUFFER_SIZE_KB / kb_per_min if kb_per_min > 0 else float('inf')
    seconds_in_queue = (queue_size_kb / kb_per_sec) if kb_per_sec > 0 else float('inf')
    total_kb_per_min += kb_per_min
    min_str = f"{minutes_in_buffer:.2f}" if minutes_in_buffer != float('inf') else "inf"
    sec_queue_str = f"{seconds_in_queue:.2f}" if seconds_in_queue != float('inf') else "inf"
    print(f"{k:<36} {v['min']/1024:>12.2f} {v['max']/1024:>12.2f} {avg/1024:>12.2f} {kb_per_min:>12.2f} {kb_per_sec:>12.2f} {min_str:>18} {sec_queue_str:>18}")

  # Summary section
  print()
  print(f"Total usage: {total_kb_per_min / 1024:.2f} MB/min")

  # Calculate memory usage: old (10MB for all) vs new (from services.py)
  OLD_SIZE = 10 * 1024 * 1024  # 10MB was the old default
  old_total = len(SERVICE_LIST) * OLD_SIZE

  new_total = sum(s.queue_size for s in SERVICE_LIST.values())

  # Count by queue size
  size_counts = {QueueSize.BIG: 0, QueueSize.MEDIUM: 0, QueueSize.SMALL: 0}
  for s in SERVICE_LIST.values():
    size_counts[s.queue_size] += 1

  savings_pct = (1 - new_total / old_total) * 100

  print()
  print(f"{'Queue Size Comparison':<40}")
  print("-" * 60)
  print(f"{'Old (10MB default):':<30} {old_total / 1024 / 1024:>10.2f} MB")
  print(f"{'New (from services.py):':<30} {new_total / 1024 / 1024:>10.2f} MB")
  print(f"{'Savings:':<30} {savings_pct:>10.1f}%")
  print()
  print(f"{'Breakdown:':<30}")
  print(f"  BIG (10MB):    {size_counts[QueueSize.BIG]:>3} services")
  print(f"  MEDIUM (2MB):  {size_counts[QueueSize.MEDIUM]:>3} services")
  print(f"  SMALL (250KB): {size_counts[QueueSize.SMALL]:>3} services")
