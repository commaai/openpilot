#!/usr/bin/env python3
import sys
import math
import capnp
import numbers
import dictdiffer
from collections import Counter

from tools.lib.logreader import LogReader

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


def get_field_tolerance(diff_field, field_tolerances):
  diff_field_str = diff_field[0]
  for s in diff_field[1:]:
    # loop until number in field
    if not isinstance(s, str):
      break
    diff_field_str += '.'+s
  if diff_field_str in field_tolerances:
    return field_tolerances[diff_field_str]


def compare_logs(log1, log2, ignore_fields=None, ignore_msgs=None, tolerance=None, field_tolerances=None):
  if ignore_fields is None:
    ignore_fields = []
  if ignore_msgs is None:
    ignore_msgs = []
  if field_tolerances is None:
    field_tolerances = {}
  default_tolerance = EPSILON if tolerance is None else tolerance

  log1, log2 = (
    [m for m in log if m.which() not in ignore_msgs]
    for log in (log1, log2)
  )

  if len(log1) != len(log2):
    cnt1 = Counter(m.which() for m in log1)
    cnt2 = Counter(m.which() for m in log2)
    raise Exception(f"logs are not same length: {len(log1)} VS {len(log2)}\n\t\t{cnt1}\n\t\t{cnt2}")

  diff = []
  for msg1, msg2 in zip(log1, log2):
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
            field_tolerance = default_tolerance
            if (tol := get_field_tolerance(diff[1], field_tolerances)) is not None:
              field_tolerance = tol
            a, b = diff[2]
            finite = math.isfinite(a) and math.isfinite(b)
            if finite and isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
              return abs(a - b) > max(field_tolerance, field_tolerance * max(abs(a), abs(b)))
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
