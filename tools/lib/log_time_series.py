import numpy as np


def flatten_type_dict(d, sep="/", prefix=None):
  res = {}
  if isinstance(d, dict):
    for key, val in d.items():
      if prefix is None:
        res.update(flatten_type_dict(val, prefix=key))
      else:
        res.update(flatten_type_dict(val, prefix=prefix + sep + key))
    return res
  elif isinstance(d, list):
    return {prefix: np.array(d)}
  else:
    return {prefix: d}


def get_message_dict(message, typ):
  valid = message.valid
  message = message._get(typ)
  if not hasattr(message, 'to_dict') or typ in ('qcomGnss', 'ubloxGnss'):
    # TODO: support these
    #print("skipping", typ)
    return

  msg_dict = message.to_dict(verbose=True)
  msg_dict = flatten_type_dict(msg_dict)
  msg_dict['_valid'] = valid
  return msg_dict


def append_dict(path, t, d, values):
  if path not in values:
    group = {}
    group["t"] = []
    for k in d:
      group[k] = []
    values[path] = group
  else:
    group = values[path]

  group["t"].append(t)
  for k, v in d.items():
    group[k].append(v)


def potentially_ragged_array(arr, dtype=None, **kwargs):
  # TODO: is there a better way to detect inhomogeneous shapes?
  try:
    return np.array(arr, dtype=dtype, **kwargs)
  except ValueError:
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

    tm = msg.logMonoTime / 1.0e9
    msg_dict = get_message_dict(msg, typ)
    if msg_dict is not None:
      append_dict(typ, tm, msg_dict, values)

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
  print(m['driverCameraState']['t'])
  print(np.diff(m['driverCameraState']['timestampSof']))
