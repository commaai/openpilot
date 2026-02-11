#!/usr/bin/env python3
import argparse
import os
from collections import defaultdict

import numpy as np
from tabulate import tabulate

from openpilot.tools.lib.logreader import LogReader

DEMO_ROUTE = "a2a0ccea32023010|2023-07-27--13-01-19"
MB = 1024 * 1024
TABULATE_OPTS = dict(tablefmt="simple_grid", stralign="center", numalign="center")


def _get_procs():
  from openpilot.selfdrive.test.test_onroad import PROCS
  return PROCS


def is_openpilot_proc(name):
  if any(p in name for p in _get_procs()):
    return True
  # catch openpilot processes not in PROCS (athenad, manager, etc.)
  return 'openpilot' in name or name.startswith(('selfdrive.', 'system.'))


def get_proc_name(proc):
  if len(proc.cmdline) > 0:
    return list(proc.cmdline)[0]
  return proc.name


def pct(val_mb, total_mb):
  return val_mb / total_mb * 100 if total_mb else 0


def has_pss(proc_logs):
  """Check if logs contain PSS data (new field, not in old logs)."""
  try:
    for proc in proc_logs[-1].procLog.procs:
      if proc.memPss > 0:
        return True
  except AttributeError:
    pass
  return False


def print_summary(proc_logs, device_states):
  mem = proc_logs[-1].procLog.mem
  total = mem.total / MB
  used = (mem.total - mem.available) / MB
  cached = mem.cached / MB
  shared = mem.shared / MB
  buffers = mem.buffers / MB

  lines = [
    f"  Total: {total:.0f} MB",
    f"  Used (total-avail): {used:.0f} MB ({pct(used, total):.0f}%)",
    f"  Cached: {cached:.0f} MB ({pct(cached, total):.0f}%)    Buffers: {buffers:.0f} MB ({pct(buffers, total):.0f}%)",
    f"  Shared/MSGQ: {shared:.0f} MB ({pct(shared, total):.0f}%)",
  ]

  if device_states:
    mem_pcts = [m.deviceState.memoryUsagePercent for m in device_states]
    lines.append(f"  deviceState memory: {np.min(mem_pcts)}-{np.max(mem_pcts)}% (avg {np.mean(mem_pcts):.0f}%)")

  print("\n-- Memory Summary --")
  print("\n".join(lines))
  return total


def collect_per_process_mem(proc_logs, use_pss):
  """Collect per-process memory samples. Returns {name: {metric: [values_per_sample_in_MB]}}."""
  by_proc = defaultdict(lambda: defaultdict(list))

  for msg in proc_logs:
    sample = defaultdict(lambda: defaultdict(float))
    for proc in msg.procLog.procs:
      name = get_proc_name(proc)
      sample[name]['rss'] += proc.memRss / MB
      if use_pss:
        sample[name]['pss'] += proc.memPss / MB
        sample[name]['pss_anon'] += proc.memPssAnon / MB
        sample[name]['pss_shmem'] += proc.memPssShmem / MB

    for name, metrics in sample.items():
      for metric, val in metrics.items():
        by_proc[name][metric].append(val)

  return by_proc


def _has_pss_detail(by_proc) -> bool:
  """Check if any process has non-zero pss_anon/pss_shmem (unavailable on some kernels)."""
  return any(sum(v.get('pss_anon', [])) > 0 or sum(v.get('pss_shmem', [])) > 0 for v in by_proc.values())


def process_table_rows(by_proc, total_mb, use_pss, show_detail):
  """Build table rows. Returns (rows, total_row)."""
  mem_key = 'pss' if use_pss else 'rss'
  rows = []
  for name in sorted(by_proc, key=lambda n: np.mean(by_proc[n][mem_key]), reverse=True):
    m = by_proc[name]
    vals = m[mem_key]
    avg = round(np.mean(vals))
    row = [name, f"{avg} MB", f"{round(np.max(vals))} MB", f"{round(pct(avg, total_mb), 1)}%"]
    if show_detail:
      row.append(f"{round(np.mean(m['pss_anon']))} MB")
      row.append(f"{round(np.mean(m['pss_shmem']))} MB")
    rows.append(row)

  # Total row
  total_row = None
  if by_proc:
    max_samples = max(len(v[mem_key]) for v in by_proc.values())
    totals = []
    for i in range(max_samples):
      s = sum(v[mem_key][i] for v in by_proc.values() if i < len(v[mem_key]))
      totals.append(s)
    avg_total = round(np.mean(totals))
    total_row = ["TOTAL", f"{avg_total} MB", f"{round(np.max(totals))} MB", f"{round(pct(avg_total, total_mb), 1)}%"]
    if show_detail:
      total_row.append(f"{round(sum(np.mean(v['pss_anon']) for v in by_proc.values()))} MB")
      total_row.append(f"{round(sum(np.mean(v['pss_shmem']) for v in by_proc.values()))} MB")

  return rows, total_row


def print_process_tables(op_procs, other_procs, total_mb, use_pss):
  all_procs = {**op_procs, **other_procs}
  show_detail = use_pss and _has_pss_detail(all_procs)

  header = ["process", "avg", "max", "%"]
  if show_detail:
    header += ["anon", "shmem"]

  op_rows, op_total = process_table_rows(op_procs, total_mb, use_pss, show_detail)
  # filter other: >5MB avg and not bare interpreter paths (test infra noise)
  other_filtered = {n: v for n, v in other_procs.items()
                    if np.mean(v['pss' if use_pss else 'rss']) > 5.0
                    and os.path.basename(n.split()[0]) not in ('python', 'python3')}
  other_rows, other_total = process_table_rows(other_filtered, total_mb, use_pss, show_detail)

  rows = op_rows
  if op_total:
    rows.append(op_total)
  if other_rows:
    sep_width = len(header)
    rows.append([""] * sep_width)
    rows.extend(other_rows)
    if other_total:
      other_total[0] = "TOTAL (other)"
      rows.append(other_total)

  metric = "PSS (no shared double-count)" if use_pss else "RSS (includes shared, overcounts)"
  print(f"\n-- Per-Process Memory: {metric} --")
  print(tabulate(rows, header, **TABULATE_OPTS))


def print_memory_accounting(proc_logs, op_procs, other_procs, total_mb, use_pss):
  last = proc_logs[-1].procLog.mem
  used = (last.total - last.available) / MB
  shared = last.shared / MB
  cached_buf = (last.buffers + last.cached) / MB - shared  # shared (MSGQ) is in Cached; separate it
  msgq = shared

  mem_key = 'pss' if use_pss else 'rss'
  op_total = sum(v[mem_key][-1] for v in op_procs.values()) if op_procs else 0
  other_total = sum(v[mem_key][-1] for v in other_procs.values()) if other_procs else 0
  proc_sum = op_total + other_total
  remainder = used - (cached_buf + msgq) - proc_sum

  if not use_pss:
    # RSS double-counts shared; add back once to partially correct
    remainder += shared

  header = ["", "MB", "%", ""]
  label = "PSS" if use_pss else "RSS*"
  rows = [
    ["Used (total - avail)", f"{used:.0f}", f"{pct(used, total_mb):.1f}", "memory in use by the system"],
    ["  Cached + Buffers", f"{cached_buf:.0f}", f"{pct(cached_buf, total_mb):.1f}", "pagecache + fs metadata, reclaimable"],
    ["  MSGQ (shared)", f"{msgq:.0f}", f"{pct(msgq, total_mb):.1f}", "/dev/shm tmpfs, also in process PSS"],
    [f"  openpilot {label}", f"{op_total:.0f}", f"{pct(op_total, total_mb):.1f}", "sum of openpilot process memory"],
    [f"  other {label}", f"{other_total:.0f}", f"{pct(other_total, total_mb):.1f}", "sum of non-openpilot process memory"],
    ["  kernel/ION/GPU", f"{remainder:.0f}", f"{pct(remainder, total_mb):.1f}", "slab, ION/DMA-BUF, GPU, page tables"],
  ]
  note = "" if use_pss else " (*RSS overcounts shared mem)"
  print(f"\n-- Memory Accounting (last sample){note} --")
  print(tabulate(rows, header, tablefmt="simple_grid", stralign="right"))


def print_report(proc_logs, device_states=None):
  """Print full memory analysis report. Can be called from tests or CLI."""
  if not proc_logs:
    print("No procLog messages found")
    return

  print(f"{len(proc_logs)} procLog samples, {len(device_states or [])} deviceState samples")

  use_pss = has_pss(proc_logs)
  if not use_pss:
    print("  (no PSS data — re-record with updated proclogd for accurate numbers)")

  total_mb = print_summary(proc_logs, device_states or [])

  by_proc = collect_per_process_mem(proc_logs, use_pss)
  op_procs = {n: v for n, v in by_proc.items() if is_openpilot_proc(n)}
  other_procs = {n: v for n, v in by_proc.items() if not is_openpilot_proc(n)}

  print_process_tables(op_procs, other_procs, total_mb, use_pss)
  print_memory_accounting(proc_logs, op_procs, other_procs, total_mb, use_pss)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Analyze memory usage from route logs")
  parser.add_argument("route", nargs="?", default=None, help="route ID or local rlog path")
  parser.add_argument("--demo", action="store_true", help=f"use demo route ({DEMO_ROUTE})")
  args = parser.parse_args()

  if args.demo:
    route = DEMO_ROUTE
  elif args.route:
    route = args.route
  else:
    parser.error("provide a route or use --demo")

  print(f"Reading logs from: {route}")

  proc_logs = []
  device_states = []
  for msg in LogReader(route):
    if msg.which() == 'procLog':
      proc_logs.append(msg)
    elif msg.which() == 'deviceState':
      device_states.append(msg)

  print_report(proc_logs, device_states)
