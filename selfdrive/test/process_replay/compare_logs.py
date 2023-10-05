#!/usr/bin/env python3
import sys
import math
import capnp
import numbers
import dictdiffer
from collections import defaultdict

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

  msgs_by_which_log1 = defaultdict(list)
  msgs_by_which_log2 = defaultdict(list)

  for msg1 in log1:
    msgs_by_which_log1[msg1.which()].append(msg1)
  for msg2 in log2:
    msgs_by_which_log2[msg2.which()].append(msg2)

  if set(msgs_by_which_log1) != set(msgs_by_which_log2):
    raise Exception(f"logs service keys don't match:\n\t\t{set(msgs_by_which_log1)}\n\t\t{set(msgs_by_which_log2)}")

  diff = []
  for which in msgs_by_which_log1.keys():
    if len(msgs_by_which_log1[which]) != len(msgs_by_which_log2[which]):
      # Print new/removed messages
      dict_msgs1 = [remove_ignored_fields(msg1, ignore_fields).as_reader().to_dict(verbose=True) for msg1 in msgs_by_which_log1[which]]
      dict_msgs2 = [remove_ignored_fields(msg2, ignore_fields).as_reader().to_dict(verbose=True) for msg2 in msgs_by_which_log2[which]]
      diff.extend(list(dictdiffer.diff(dict_msgs1, dict_msgs2)))
    else:
      for msg1, msg2 in zip(msgs_by_which_log1[which], msgs_by_which_log2[which], strict=True):
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


if __name__ == "__main__":
  log1 = list(LogReader(sys.argv[1]))
  log2 = list(LogReader(sys.argv[2]))
  print(compare_logs(log1, log2, sys.argv[3:]))
