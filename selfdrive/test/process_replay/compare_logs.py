#!/usr/bin/env python3
import sys
import math
import capnp
import numbers
import dictdiffer
from collections import Counter

from openpilot.tools.lib.logreader import LogReader

EPSILON = sys.float_info.epsilon


def remove_ignored_fields(msg, ignore):
  msg = msg.as_builder()
  for key in ignore:
    attr = msg
    keys = key.split(".")
    if msg.which() != keys[0] and len(keys) > 1:
      continue

    for k in keys[:-1]:
      # indexing into list
      if k.isdigit():
        attr = attr[int(k)]
      else:
        attr = getattr(attr, k)

    v = getattr(attr, keys[-1])
    if isinstance(v, bool):
      val = False
    elif isinstance(v, numbers.Number):
      val = 0
    elif isinstance(v, (list, capnp.lib.capnp._DynamicListBuilder)):
      val = []
    else:
      raise NotImplementedError(f"Unknown type: {type(v)}")
    setattr(attr, keys[-1], val)
  return msg


def compare_logs(log1, log2, ignore_fields=None, ignore_msgs=None, tolerance=None,):
  if ignore_fields is None:
    ignore_fields = []
  if ignore_msgs is None:
    ignore_msgs = []
  tolerance = EPSILON if tolerance is None else tolerance

  log1, log2 = (
    [m for m in log if m.which() not in ignore_msgs]
    for log in (log1, log2)
  )

  if len(log1) != len(log2):
    cnt1 = Counter(m.which() for m in log1)
    cnt2 = Counter(m.which() for m in log2)
    raise Exception(f"logs are not same length: {len(log1)} VS {len(log2)}\n\t\t{cnt1}\n\t\t{cnt2}")

  diff = []
  for msg1, msg2 in zip(log1, log2, strict=True):
    if msg1.which() != msg2.which():
      raise Exception("msgs not aligned between logs")

    msg1 = remove_ignored_fields(msg1, ignore_fields)
    msg2 = remove_ignored_fields(msg2, ignore_fields)

    if msg1.to_bytes() != msg2.to_bytes():
      msg1_dict = msg1.as_reader().to_dict(verbose=True)
      msg2_dict = msg2.as_reader().to_dict(verbose=True)

      dd = dictdiffer.diff(msg1_dict, msg2_dict, ignore=ignore_fields)

      # Dictdiffer only supports relative tolerance, we also want to check for absolute
      # TODO: add this to dictdiffer
      def outside_tolerance(diff):
        try:
          if diff[0] == "change":
            a, b = diff[2]
            finite = math.isfinite(a) and math.isfinite(b)
            if finite and isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
              return abs(a - b) > max(tolerance, tolerance * max(abs(a), abs(b)))
        except TypeError:
          pass
        return True

      dd = list(filter(outside_tolerance, dd))

      diff.extend(dd)
  return diff


def format_process_diff(diff):
  diff_short, diff_long = "", ""

  if isinstance(diff, str):
    diff_short += f"        {diff}\n"
    diff_long += f"\t{diff}\n"
  else:
    cnt: dict[str, int] = {}
    for d in diff:
      diff_long += f"\t{str(d)}\n"

      k = str(d[1])
      cnt[k] = 1 if k not in cnt else cnt[k] + 1

    for k, v in sorted(cnt.items()):
      diff_short += f"        {k}: {v}\n"

  return diff_short, diff_long


def format_diff(results, log_paths, ref_commit):
  diff_short, diff_long = "", ""
  diff_long += f"***** tested against commit {ref_commit} *****\n"

  failed = False
  for segment, result in list(results.items()):
    diff_short += f"***** results for segment {segment} *****\n"
    diff_long += f"***** differences for segment {segment} *****\n"

    for proc, diff in list(result.items()):
      diff_long += f"*** process: {proc} ***\n"
      diff_long += f"\tref: {log_paths[segment][proc]['ref']}\n"
      diff_long += f"\tnew: {log_paths[segment][proc]['new']}\n\n"

      diff_short += f"    {proc}\n"

      if isinstance(diff, str) or len(diff):
        diff_short += f"        ref: {log_paths[segment][proc]['ref']}\n"
        diff_short += f"        new: {log_paths[segment][proc]['new']}\n\n"
        failed = True

        proc_diff_short, proc_diff_long = format_process_diff(diff)

        diff_long += proc_diff_long
        diff_short += proc_diff_short

  return diff_short, diff_long, failed


if __name__ == "__main__":
  log1 = list(LogReader(sys.argv[1]))
  log2 = list(LogReader(sys.argv[2]))
  ignore_fields = sys.argv[3:] or ["logMonoTime"]
  results = {"segment": {"proc": compare_logs(log1, log2, ignore_fields)}}
  log_paths = {"segment": {"proc": {"ref": sys.argv[1], "new": sys.argv[2]}}}
  diff_short, diff_long, failed = format_diff(results, log_paths, None)

  print(diff_long)
  print(diff_short)
