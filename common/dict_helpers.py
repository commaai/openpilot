# remove all keys that end in DEPRECATED
def strip_deprecated_keys(d):
  for k in list(d.keys()):
    if type(k) is str:
      if k.endswith('DEPRECATED'):
        d.pop(k)
      elif type(d[k]) is dict:
        strip_deprecated_keys(d[k])
  return d
