import os
import errno
from typing import Any

try:
  # Tell mypy to ignore these lines since these functions may not exist
  os_getxattr = os.getxattr  # type: ignore
  os_setxattr = os.setxattr  # type: ignore
except AttributeError:
  # Fallback implementations if os doesn't support xattr
  def os_getxattr(path: str, attr_name: str) -> bytes:
    raise OSError(errno.ENOTSUP, "xattr not supported")

  def os_setxattr(path: str, attr_name: str, attr_value: bytes) -> None:
    raise OSError(errno.ENOTSUP, "xattr not supported")

_cached_attributes: dict[tuple[str, str], bytes | None] = {}

def getxattr(path: str, attr_name: str) -> bytes | None:
  key = (path, attr_name)
  if key not in _cached_attributes:
    try:
      response = os_getxattr(path, attr_name)
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
  os_setxattr(path, attr_name, attr_value)
