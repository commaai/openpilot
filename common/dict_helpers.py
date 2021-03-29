# remove all keys that end in DEPRECATED
def strip_deprecated_keys(d):
  for k in list(d.keys()):
    if isinstance(k, str):
      if k.endswith('DEPRECATED'):
        d.pop(k)
      elif isinstance(d[k], dict):
        strip_deprecated_keys(d[k])
  return d
