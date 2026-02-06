import os
from collections import defaultdict

from opendbc.car.tests.car_diff import format_diff as format_car_diff
from openpilot.selfdrive.test.process_replay.compare_logs import compare_logs
from openpilot.selfdrive.test.process_replay.process_replay import PROC_REPLAY_DIR

NAN_FIELDS = {'aRel', 'yvRel'}


class MsgWrap:
  """Adapter so to_dict() includes defaults"""
  def __init__(self, msg):
    self._msg = msg
  def to_dict(self):
    return self._msg.to_dict(verbose=True)


def diff_process(cfg, ref_msgs, new_msgs):
  ref = defaultdict(list)
  new = defaultdict(list)
  for m in ref_msgs:
    if m.which() in cfg.subs:
      ref[m.which()].append(m)
  for m in new_msgs:
    if m.which() in cfg.subs:
      new[m.which()].append(m)

  diffs = []
  for sub in cfg.subs:
    if len(ref[sub]) != len(new[sub]):
      diffs.append((f"{sub} (message count)", 0, (len(ref[sub]), len(new[sub])), 0))
    for i, (r, n) in enumerate(zip(ref[sub], new[sub], strict=False)):
      for d in compare_logs([r], [n], cfg.ignore, tolerance=cfg.tolerance):
        if d[0] == "change":
          path = ".".join(str(p) for p in d[1]) if isinstance(d[1], list) else d[1]
          if cfg.proc_name == "card" and path.split('.')[-1] in NAN_FIELDS:
            continue
          diffs.append((path, i, d[2], r.logMonoTime))
        elif d[0] in ("add", "remove"):
          path = ".".join(str(p) for p in d[1]) if isinstance(d[1], list) else d[1]
          if cfg.proc_name == "card" and path.split('.')[-1] in NAN_FIELDS:
            continue
          for item in d[2]:
            diffs.append((f"{path}.{item[0]}", i, (d[0], item[1]), r.logMonoTime))
  return (diffs, ref, new) if diffs else None


def diff_format(diffs, ref, new, field):
  msg_type = field.split(".")[0]
  ref_ts = [(m.logMonoTime, MsgWrap(m)) for m in ref.get(msg_type, [])]
  new_wrapped = [MsgWrap(m) for m in new.get(msg_type, [])]
  return format_car_diff(diffs, ref_ts, new_wrapped, field)


def diff_report(replay_diffs, segments):
  seg_to_plat = {seg: plat for plat, seg in segments}

  with_diffs, errors, n_passed = [], [], 0
  for seg, proc, data in replay_diffs:
    plat = seg_to_plat.get(seg, "UNKNOWN")
    if data is None:
      n_passed += 1
    elif isinstance(data, str):
      errors.append((plat, seg, proc, data))
    else:
      with_diffs.append((plat, seg, proc, data))

  icon = "⚠️" if with_diffs else "✅"
  lines = [
    "## Process replay diff report",
    "Replays driving segments through this PR and compares the behavior to master.",
    "Please review any changes carefully to ensure they are expected.\n",
    f"{icon}  {len(with_diffs)} changed, {n_passed} passed, {len(errors)} errors",
  ]

  for plat, seg, proc, err in errors:
    lines.append(f"\nERROR {plat} - {seg} [{proc}]: {err}")

  if with_diffs:
    lines.append("<details><summary><b>Show changes</b></summary>\n\n```")
    for plat, seg, proc, (diffs, ref, new) in with_diffs:
      lines.append(f"\n{plat} - {seg} [{proc}]")
      by_field = defaultdict(list)
      for d in diffs:
        by_field[d[0]].append(d)
      for field, fd in sorted(by_field.items()):
        lines.append(f"\n  {field} ({len(fd)} diffs)")
        lines.extend(diff_format(fd, ref, new, field))
    lines.append("```\n</details>")

  with open(os.path.join(PROC_REPLAY_DIR, "diff_report.txt"), "w") as f:
    f.write("\n".join(lines))
