import collections


def create_namedtuple_from_dict(obj):
  if isinstance(obj, dict):
    fields = sorted(obj.keys())
    namedtuple_type = collections.namedtuple(typename='OPNamedType', field_names=fields, rename=True, defaults=(None,))
    field_value_pairs = collections.OrderedDict((str(field), create_namedtuple_from_dict(obj[field])) for field in fields)
    return namedtuple_type(**field_value_pairs)
  elif isinstance(obj, (list, set, tuple, frozenset)):
    return [create_namedtuple_from_dict(item) for item in obj]
  else:
    return obj


def create_namedtuple_from_capnp(reader):
  d = reader.to_dict(verbose=True, ordered=True)
  return create_namedtuple_from_dict(d)
