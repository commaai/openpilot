import os
import sys
from collections import defaultdict

from opendbc.car.tests.car_diff import format_diff
from openpilot.common.git import get_commit
from openpilot.selfdrive.test.process_replay.compare_logs import compare_logs
from openpilot.selfdrive.test.process_replay.process_replay import CONFIGS, FAKEDATA, replay_process
from openpilot.selfdrive.test.process_replay.test_processes import segments, get_log_data, REF_COMMIT_FN, EXCLUDED_PROCS, BASE_URL
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.url_file import URLFile

NAN_FIELDS = {'aRel', 'yvRel'}


class MsgWrap:
  """Adapter so to_dict() includes defaults"""
  def __init__(self, msg):
    self._msg = msg
  def to_dict(self):
    return self._msg.to_dict(verbose=True)


def compare_proc(cfg, ref_msgs, new_msgs):
  ref, new = defaultdict(list), defaultdict(list)
  for m in ref_msgs:
    if m.which() in cfg.subs:
      ref[m.which()].append(m)
  for m in new_msgs:
    if m.which() in cfg.subs:
      new[m.which()].append(m)

  diffs = []
  for sub in cfg.subs:
    for i, (r, n) in enumerate(zip(ref[sub], new[sub], strict=True)):
      for d in compare_logs([r], [n], cfg.ignore, tolerance=cfg.tolerance):
        if d[0] == "change":
          path = ".".join(str(p) for p in d[1]) if isinstance(d[1], list) else d[1]
          if cfg.proc_name == "card" and path.split('.')[-1] in NAN_FIELDS:
            continue
          diffs.append((path, i, d[2], r.logMonoTime))
  return diffs, ref, new


def format_proc(diffs, ref, new, field):
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

  print("## Process replay diff report")
  print("Replays driving segments through this PR and compares the behavior to master.")
  print("Please review any changes carefully to ensure they are expected.\n")

  results = []
  for plat, seg in segments:
    lr_data = None
    for cfg in CONFIGS:
      if cfg.proc_name in EXCLUDED_PROCS:
        continue
      if cfg.proc_name not in ('card', 'controlsd', 'lagd') and plat not in ('HYUNDAI', 'TOYOTA'):
        continue

      cur_log_fn = os.path.join(FAKEDATA, f"{seg}_{cfg.proc_name}_{cur_commit}.zst".replace("|", "_"))
      ref_fn = os.path.join(FAKEDATA, f"{seg}_{cfg.proc_name}_{ref_commit}.zst".replace("|", "_"))
      ref_path = ref_fn if os.path.exists(ref_fn) else BASE_URL + os.path.basename(ref_fn)
      try:
        if os.path.exists(cur_log_fn):
          new_msgs = list(LogReader(cur_log_fn))
        else:
          if lr_data is None:
            _, lr_data = get_log_data(seg)
          new_msgs = replay_process(cfg, LogReader.from_bytes(lr_data), disable_progress=True)
        diffs, ref, new = compare_proc(cfg, list(LogReader(ref_path)), new_msgs)
        results.append((plat, seg, cfg.proc_name, (diffs, ref, new) if diffs else None, None))
      except Exception as e:
        results.append((plat, seg, cfg.proc_name, None, str(e)))

  with_diffs = [(plat, seg, proc, res) for plat, seg, proc, res, err in results if res]
  errors = [(plat, seg, proc, err) for plat, seg, proc, _, err in results if err]
  n_passed = len(results) - len(with_diffs) - len(errors)

  icon = "⚠️" if with_diffs else "✅"
  print(f"{icon}  {len(with_diffs)} changed, {n_passed} passed, {len(errors)} errors")

  for plat, seg, proc, err in errors:
    print(f"\nERROR {plat} - {seg} [{proc}]: {err}")

  if with_diffs:
    print("<details><summary><b>Show changes</b></summary>\n\n```")
    for plat, seg, proc, (diffs, ref, new) in with_diffs:
      print(f"\n{plat} - {seg} [{proc}]")
      by_field = defaultdict(list)
      for d in diffs:
        by_field[d[0]].append(d)
      for field, fd in sorted(by_field.items()):
        print(f"\n  {field} ({len(fd)} diffs)")
        for line in format_proc(fd, ref, new, field):
          print(line)
    print("```\n</details>")

  return 1 if errors else 0


if __name__ == "__main__":
  sys.exit(main())
