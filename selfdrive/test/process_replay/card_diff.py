#!/usr/bin/env python3
import os
import sys
from collections import defaultdict

from opendbc.car.tests.car_diff import format_diff
from openpilot.common.git import get_commit
from openpilot.selfdrive.test.process_replay.compare_logs import compare_logs
from openpilot.selfdrive.test.process_replay.process_replay import FAKEDATA, get_process_config, replay_process
from openpilot.selfdrive.test.process_replay.test_processes import segments, get_log_data, REF_COMMIT_FN
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.url_file import URLFile

BASE_URL = "https://raw.githubusercontent.com/commaai/ci-artifacts/refs/heads/process-replay/"
CARD_CFG = get_process_config("card")
NAN_FIELDS = {'aRel', 'yvRel'}


class MsgWrap:
  """Adapter so to_dict() includes defaults"""
  def __init__(self, msg):
    self._msg = msg
  def to_dict(self):
    return self._msg.to_dict(verbose=True)


def compare_card(ref_msgs, new_msgs):
  ref, new = defaultdict(list), defaultdict(list)
  for m in ref_msgs:
    if m.which() in CARD_CFG.subs:
      ref[m.which()].append(m)
  for m in new_msgs:
    if m.which() in CARD_CFG.subs:
      new[m.which()].append(m)

  diffs = []
  for sub in CARD_CFG.subs:
    for i, (r, n) in enumerate(zip(ref[sub], new[sub], strict=True)):
      for d in compare_logs([r], [n], CARD_CFG.ignore, tolerance=CARD_CFG.tolerance):
        if d[0] == "change":
          path = ".".join(str(p) for p in d[1]) if isinstance(d[1], list) else d[1]
          if path.split('.')[-1] in NAN_FIELDS:
            continue
          diffs.append((path, i, d[2], r.logMonoTime))
  return diffs, ref, new


def format_card(diffs, ref, new, field):
  msg_type = field.split(".")[0]
  ref = [(m.logMonoTime, MsgWrap(m)) for m in ref.get(msg_type, [])]
  states = [MsgWrap(m) for m in new.get(msg_type, [])]
  return format_diff(diffs, ref, states, field)


def main() -> int:
  try:
    with open(REF_COMMIT_FN) as f:
      ref_commit = f.read().strip()
  except FileNotFoundError:
    ref_commit = URLFile(BASE_URL + "ref_commit", cache=False).read().decode().strip()

  cur_commit = get_commit()
  if not cur_commit:
    raise Exception("Couldn't get current commit")

  print("## Card behavior report")
  print("Replays driving segments through this PR and compares the behavior to master.")
  print("Please review any changes carefully to ensure they are expected.\n")

  results = []
  for plat, seg in segments:
    cur_log_fn = os.path.join(FAKEDATA, f"{seg}_{CARD_CFG.proc_name}_{cur_commit}.zst".replace("|", "_"))
    ref_fn = os.path.join(FAKEDATA, f"{seg}_{CARD_CFG.proc_name}_{ref_commit}.zst".replace("|", "_"))
    ref_path = ref_fn if os.path.exists(ref_fn) else BASE_URL + os.path.basename(ref_fn)
    try:
      if os.path.exists(cur_log_fn):
        new_msgs = list(LogReader(cur_log_fn))
      else:
        _, lr_dat = get_log_data(seg)
        new_msgs = replay_process(CARD_CFG, LogReader.from_bytes(lr_dat), disable_progress=True)
      diffs, ref, new = compare_card(list(LogReader(ref_path)), new_msgs)
      if diffs:
        results.append((plat, seg, (diffs, ref, new), None))
      else:
        results.append((plat, seg, None, None))
    except Exception as e:
      results.append((plat, seg, None, str(e)))

  with_diffs = [(plat, seg, res) for plat, seg, res, err in results if res]
  errors = [(plat, seg, err) for plat, seg, _, err in results if err]
  n_passed = len(results) - len(with_diffs) - len(errors)

  icon = "⚠️" if with_diffs else "✅"
  print(f"{icon}  {len(with_diffs) - len(errors)} changed, {n_passed} passed, {len(errors)} errors")

  for plat, seg, err in errors:
    print(f"\nERROR {plat} - {seg}: {err}")

  if with_diffs:
    print("<details><summary><b>Show changes</b></summary>\n\n```")
    for plat, seg, (diffs, ref, new) in with_diffs:
      print(f"\n{plat} - {seg}")
      by_field = defaultdict(list)
      for d in diffs:
        by_field[d[0]].append(d)
      for field, fd in sorted(by_field.items()):
        print(f"\n  {field} ({len(fd)} diffs)")
        for line in format_card(fd, ref, new, field):
          print(line)
    print("```\n</details>")

  return 1 if errors else 0


if __name__ == "__main__":
  sys.exit(main())
