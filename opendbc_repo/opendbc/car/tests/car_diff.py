#!/usr/bin/env python3
import argparse
import os
import pickle
import re
import subprocess
import sys
import tempfile
import traceback
import zstandard as zstd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from urllib.request import urlopen
from collections import defaultdict
from pathlib import Path

from comma_car_segments import get_comma_car_segments_database, get_url

from opendbc.car import structs
from opendbc.car.can_definitions import CanData
from opendbc.car.car_helpers import can_fingerprint, interfaces
from opendbc.car.logreader import LogReader, decompress_stream


TOLERANCE = 1e-4
DIFF_BUCKET = "car_diff"
IGNORE_FIELDS = ["cumLagMs", "canErrorCounter"]


def dict_diff(d1, d2, path="", ignore=None, tolerance=0):
  ignore = ignore or []
  diffs = []
  for key in d1.keys() | d2.keys():
    if key in ignore:
      continue
    full_path = f"{path}.{key}" if path else key
    v1, v2 = d1.get(key), d2.get(key)
    if isinstance(v1, dict) and isinstance(v2, dict):
      diffs.extend(dict_diff(v1, v2, full_path, ignore, tolerance))
    elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
      if abs(v1 - v2) > tolerance:
        diffs.append(("change", full_path, (v1, v2)))
    elif v1 != v2:
      diffs.append(("change", full_path, (v1, v2)))
  return diffs


def load_can_messages(seg):
  parts = seg.split("/")
  url = get_url(f"{parts[0]}/{parts[1]}", parts[2])
  msgs = LogReader(url, only_union_types=True, sort_by_time=True)
  return [m for m in msgs if m.which() == 'can']


def replay_segment(platform, can_msgs):
  _can_msgs = ([CanData(can.address, can.dat, can.src) for can in m.can] for m in can_msgs)

  def can_recv(wait_for_one: bool = False) -> list[list[CanData]]:
    return [next(_can_msgs, [])]

  _, fingerprint = can_fingerprint(can_recv)

  CarInterface = interfaces[platform]
  CP = CarInterface.get_params(platform, fingerprint, [], False, False, False)
  CI = CarInterface(CP)
  CC = structs.CarControl().as_reader()

  states, timestamps = [], []
  for msg in can_msgs:
    frames = [CanData(c.address, c.dat, c.src) for c in msg.can]
    states.append(CI.update([(msg.logMonoTime, frames)]))
    CI.apply(CC, msg.logMonoTime)
    timestamps.append(msg.logMonoTime)
  return states, timestamps


def process_segment(args):
  platform, seg, ref_path, update = args
  try:
    can_msgs = load_can_messages(seg)
    states, timestamps = replay_segment(platform, can_msgs)
    ref_file = Path(ref_path) / f"{platform}_{seg.replace('/', '_')}.zst"

    if update:
      data = list(zip(timestamps, states, strict=True))
      ref_file.write_bytes(zstd.compress(pickle.dumps(data), 10))
      return (platform, seg, [], None)

    if not ref_file.exists():
      return (platform, seg, [], "no ref")

    ref = pickle.loads(decompress_stream(ref_file.read_bytes()))
    diffs = []
    for i, ((ts, ref_state), state) in enumerate(zip(ref, states, strict=True)):
      for diff in dict_diff(ref_state.to_dict(), state.to_dict(), ignore=IGNORE_FIELDS, tolerance=TOLERANCE):
        diffs.append((diff[1], i, diff[2], ts))
    return (platform, seg, diffs, None)
  except Exception:
    return (platform, seg, [], traceback.format_exc())


def get_changed_platforms(cwd, database, interfaces):
  git_ref = os.environ.get("GIT_REF", "origin/master")
  changed = subprocess.check_output(["git", "diff", "--name-only", f"{git_ref}...HEAD"], cwd=cwd, encoding='utf8').strip()
  brands = set()
  patterns = [r"opendbc/car/(\w+)/", r"opendbc/dbc/(\w+?)_", r"opendbc/dbc/generator/(\w+)", r"opendbc/safety/modes/(\w+?)[_.]"]
  for line in changed.splitlines():
    for pattern in patterns:
      m = re.search(pattern, line)
      if m:
        brands.add(m.group(1).lower())
  return [p for p in interfaces if any(b in p.lower() for b in brands) and p in database]


def download_refs(ref_path, platforms, segments):
  base_url = f"https://raw.githubusercontent.com/commaai/ci-artifacts/refs/heads/{DIFF_BUCKET}"
  for platform in tqdm(platforms):
    for seg in segments.get(platform, []):
      filename = f"{platform}_{seg.replace('/', '_')}.zst"
      with urlopen(f"{base_url}/{filename}") as resp:
        (Path(ref_path) / filename).write_bytes(resp.read())


def run_replay(platforms, segments, ref_path, update, workers=4):
  work = [(platform, seg, ref_path, update)
          for platform in platforms for seg in segments.get(platform, [])]
  return process_map(process_segment, work, max_workers=workers)


# ASCII waveforms helpers
def find_edges(vals, init):
  rises = []
  falls = []
  prev = init
  for i, val in enumerate(vals):
    if val and not prev:
      rises.append(i)
    if not val and prev:
      falls.append(i)
    prev = val
  return rises, falls


def render_waveform(label, vals, init):
  wave = {(False, False): "_", (True, True): "‾", (False, True): "/", (True, False): "\\"}
  line = f"  {label}:".ljust(12)
  prev = init
  for val in vals:
    line += wave[(prev, val)]
    prev = val
  if len(line) > 80:
    line = line[:80] + "..."
  return line


def format_timing(edge_type, master_edges, pr_edges, ms_per_frame):
  if not master_edges or not pr_edges:
    return None
  delta = pr_edges[0] - master_edges[0]
  if delta == 0:
    return None
  direction = "lags" if delta > 0 else "leads"
  ms = int(abs(delta) * ms_per_frame)
  return " " * 12 + f"{edge_type}: PR {direction} by {abs(delta)} frames ({ms}ms)"


def group_frames(diffs, max_gap=15):
  groups = []
  current = [diffs[0]]
  for diff in diffs[1:]:
    _, frame, _, _ = diff
    _, prev_frame, _, _ = current[-1]
    if frame <= prev_frame + max_gap:
      current.append(diff)
    else:
      groups.append(current)
      current = [diff]
  groups.append(current)
  return groups


def build_signals(group):
  _, first_frame, _, _ = group[0]
  _, last_frame, (final_master, _), _ = group[-1]
  start = max(0, first_frame - 5)
  end = last_frame + 6
  init = not final_master
  diff_at = {frame: (m, p) for _, frame, (m, p), _ in group}
  master_vals = []
  pr_vals = []
  master = init
  pr = init
  for frame in range(start, end):
    if frame in diff_at:
      master, pr = diff_at[frame]
    elif frame > last_frame:
      master = pr = final_master
    master_vals.append(master)
    pr_vals.append(pr)
  return master_vals, pr_vals, init, start, end


def format_numeric_diffs(diffs):
  lines = []
  for _, frame, (old_val, new_val), _ in diffs[:10]:
    lines.append(f"    frame {frame}: {old_val} -> {new_val}")
  if len(diffs) > 10:
    lines.append(f"    (... {len(diffs) - 10} more)")
  return lines


def format_boolean_diffs(diffs):
  _, first_frame, _, first_ts = diffs[0]
  _, last_frame, _, last_ts = diffs[-1]
  frame_time = last_frame - first_frame
  time_ms = (last_ts - first_ts) / 1e6
  ms = time_ms / frame_time if frame_time else 10.0
  lines = []
  for group in group_frames(diffs):
    master_vals, pr_vals, init, start, end = build_signals(group)
    master_rises, master_falls = find_edges(master_vals, init)
    pr_rises, pr_falls = find_edges(pr_vals, init)
    if bool(master_rises) != bool(pr_rises) or bool(master_falls) != bool(pr_falls):
      continue
    lines.append(f"\n  frames {start}-{end - 1}")
    lines.append(render_waveform("master", master_vals, init))
    lines.append(render_waveform("PR", pr_vals, init))
    for edge_type, master_edges, pr_edges in [("rise", master_rises, pr_rises), ("fall", master_falls, pr_falls)]:
      msg = format_timing(edge_type, master_edges, pr_edges, ms)
      if msg:
        lines.append(msg)
  return lines


def format_diff(diffs):
  if not diffs:
    return []
  _, _, (old, new), _ = diffs[0]
  is_bool = isinstance(old, bool) and isinstance(new, bool)
  if is_bool:
    return format_boolean_diffs(diffs)
  return format_numeric_diffs(diffs)


def main(platform=None, segments_per_platform=10, update_refs=False, all_platforms=False):
  cwd = Path(__file__).resolve().parents[3]
  ref_path = cwd / DIFF_BUCKET
  if not update_refs:
    ref_path = Path(tempfile.mkdtemp())
  ref_path.mkdir(exist_ok=True)
  database = get_comma_car_segments_database()

  if all_platforms:
    print("Running all platforms...")
    platforms = [p for p in interfaces if p in database]
  elif platform and platform in interfaces:
    platforms = [platform]
  else:
    platforms = get_changed_platforms(cwd, database, interfaces)

  if not platforms:
    print("No car changes detected")
    return 0

  segments = {p: database.get(p, [])[:segments_per_platform] for p in platforms}
  n_segments = sum(len(s) for s in segments.values())
  print(f"{'Generating' if update_refs else 'Testing'} {n_segments} segments for: {', '.join(platforms)}")

  if update_refs:
    results = run_replay(platforms, segments, ref_path, update=True)
    errors = [e for _, _, _, e in results if e]
    assert len(errors) == 0, f"Segment failures: {errors}"
    print(f"Generated {n_segments} refs to {ref_path}")
    return 0

  download_refs(ref_path, platforms, segments)
  results = run_replay(platforms, segments, ref_path, update=False)

  with_diffs = [(p, s, d) for p, s, d, e in results if d]
  errors = [(p, s, e) for p, s, d, e in results if e]
  n_passed = len(results) - len(with_diffs) - len(errors)

  print(f"\nResults: {n_passed} passed, {len(with_diffs)} with diffs, {len(errors)} errors")

  for plat, seg, err in errors:
    print(f"\nERROR {plat} - {seg}: {err}")

  if with_diffs:
    print("```")
    for plat, seg, diffs in with_diffs:
      print(f"\n{plat} - {seg}")
      by_field = defaultdict(list)
      for d in diffs:
        by_field[d[0]].append(d)
      for field, fd in sorted(by_field.items()):
        print(f"  {field} ({len(fd)} diffs)")
        for line in format_diff(fd):
          print(line)
    print("```")

  return 1 if errors else 0


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--platform", help="diff single platform")
  parser.add_argument("--segments-per-platform", type=int, default=10, help="number of segments to diff per platform")
  parser.add_argument("--update-refs", action="store_true", help="update refs based on current commit")
  parser.add_argument("--all", action="store_true", help="run diff on all platforms")
  args = parser.parse_args()
  sys.exit(main(args.platform, args.segments_per_platform, args.update_refs, args.all))
