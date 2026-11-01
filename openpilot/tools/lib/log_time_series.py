import capnp
import numpy as np
from capnp.lib.capnp import _DynamicEnum


def _list_to_python(value):
  return [x.to_dict(verbose=True) if hasattr(x, 'to_dict') else
          _list_to_python(x) if isinstance(x, (capnp._DynamicListReader, capnp._DynamicListBuilder)) else
          str(x) if isinstance(x, _DynamicEnum) else x for x in value]


class _MessageConverter:
  """Cache schema field handles and destination columns for one service.

  Read directly into columns instead of building and flattening a dictionary for
  every message. Union branches are initialized only when they are active.
  """
  def __init__(self, schema, group, prefix=""):
    self.schema = schema
    self.group = group
    self.prefix = prefix
    self.fields = [self._field(name) for name in schema.non_union_fields]
    self.union_fields = {}

  def _field(self, name):
    field = self.schema.fields[name]
    kind = 'struct' if field.proto.which() == 'group' else field.proto.slot.type.which()
    key = self.prefix + name
    if kind == 'struct':
      dest = _MessageConverter(field.schema, self.group, key + '/')
    else:
      dest = self.group[key] = []
    return field, kind, dest

  def append(self, message):
    fields = self.fields
    if self.schema.union_fields:
      name = message.which()
      if name not in self.union_fields:
        self.union_fields[name] = self._field(name)
      fields = [self.union_fields[name], *fields]
    for field, kind, dest in fields:
      value = message._get_by_field(field)
      if kind == 'struct':
        dest.append(value)
      else:
        if kind == 'list':
          value = _list_to_python(value)
        elif kind == 'enum':
          value = str(value)
        dest.append(value)


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
  converters = {}
  for msg in msgs:
    typ = msg.which()
    if typ in ('qcomGnss', 'ubloxGnss'):
      continue
    message = msg._get(typ)
    if not hasattr(message, 'to_dict'):
      continue

    if typ not in values:
      values[typ] = {"t": [], "_valid": []}
      converters[typ] = _MessageConverter(message.schema, values[typ])
    group = values[typ]
    group["t"].append(msg.logMonoTime / 1.0e9)
    group["_valid"].append(msg.valid)
    converters[typ].append(message)

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
