import numpy as np

UNSUPPORTED_TYPES = frozenset(("qcomGnss", "ubloxGnss"))

def _flatten_type_dict_into(res, d, sep="/", prefix=None):
  stack = [(prefix, d)]
  while stack:
    path, value = stack.pop()
    if isinstance(value, dict):
      if path is None:
        for key, child in value.items():
          stack.append((key, child))
      else:
        prefix = path + sep
        for key, child in value.items():
          stack.append((prefix + key, child))
    else:
      res[path] = value


def flatten_type_dict(d, sep="/", prefix=None):
  res = {}
  _flatten_type_dict_into(res, d, sep=sep, prefix=prefix)
  return res


def get_message_dict(message, typ):
  if typ in UNSUPPORTED_TYPES:
    # TODO: support these
    return

  valid = message.valid
  message = message._get(typ)
  if not hasattr(message, 'to_dict'):
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
