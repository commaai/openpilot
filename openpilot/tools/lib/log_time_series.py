import numpy as np


def _append_fields(d, group, prefix=""):
  for key, value in d.items():
    name = prefix + key
    if isinstance(value, dict):
      _append_fields(value, group, name + "/")
    else:
      if name not in group:
        group[name] = []
      group[name].append(value)


def potentially_ragged_array(arr, dtype=None, **kwargs):
  # TODO: is there a better way to detect inhomogeneous shapes?
  try:
    return np.array(arr, dtype=dtype, **kwargs)
  except ValueError:
    # Preserve array-valued rows when a list field has varying lengths.
    arr = [np.array(value) if isinstance(value, list) else value for value in arr]
    return np.array(arr, dtype=object, **kwargs)

def msgs_to_time_series(msgs):
  """
    Convert an iterable of canonical capnp messages into a dictionary of time series.
    Each time series has a value with key "t" which consists of monotonically increasing timestamps
    in seconds.
  """
  values = {}
  for msg in msgs:
    typ = msg.which()
    if typ in ('qcomGnss', 'ubloxGnss'):
      continue
    message = msg._get(typ)
    if not hasattr(message, 'to_dict'):
      continue

    if typ not in values:
      values[typ] = {"t": [], "_valid": []}
    group = values[typ]
    group["t"].append(msg.logMonoTime / 1.0e9)
    group["_valid"].append(msg.valid)
    # Flatten straight into columns; convert lists to arrays once per column below.
    _append_fields(message.to_dict(verbose=True), group)

  # Sort values by time.
  for group in values.values():
    order = np.argsort(group["t"])
    for name, group_values in group.items():
      group[name] = potentially_ragged_array(group_values)[order]

  return values


if __name__ == "__main__":
  import sys
  from openpilot.tools.lib.logreader import LogReader
  m = msgs_to_time_series(LogReader(sys.argv[1]))
  print(m['cabinCameraState']['t'])
  print(np.diff(m['cabinCameraState']['timestampSof']))
