# remove all keys that end in DEPRECATED
def strip_deprecated_keys(d):
  for k in d.keys():
    if type(k) == str and k.endswith('DEPRECATED'):
      d.pop(k)
  return d
