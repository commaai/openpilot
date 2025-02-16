import errno
import sys
if sys.platform == 'linux':
  import os
else:
  import xattr

_cached_attributes: dict[tuple, bytes | None] = {}

def getxattr(path: str, attr_name: str) -> bytes | None:
  key = (path, attr_name)
  if key not in _cached_attributes:
    try:
      if sys.platform == 'linux':
        response = os.getxattr(path, attr_name)
      else:
        response = xattr.getxattr(path, attr_name)
    except OSError as e:
      # ENODATA means attribute hasn't been set
      if e.errno == errno.ENODATA:
        response = None
      else:
        raise
    _cached_attributes[key] = response
  return _cached_attributes[key]

def setxattr(path: str, attr_name: str, attr_value: bytes) -> None:
  _cached_attributes.pop((path, attr_name), None)
  if sys.platform == 'linux':
    os.setxattr(path, attr_name, attr_value)
  else:
    xattr.setxattr(path, attr_name, attr_value)
