import os
import errno
from typing import TYPE_CHECKING
from os import PathLike
from typing_extensions import Buffer
from collections.abc import Callable

PathType = int | str | bytes | PathLike[str] | PathLike[bytes]
XAttrName = str | bytes | PathLike[str] | PathLike[bytes]

if TYPE_CHECKING:
  os_getxattr: Callable[[PathType, XAttrName, bool], bytes]
  os_setxattr: Callable[[PathType, XAttrName, Buffer, int, bool], None]

def _getxattr_fallback(path: PathType, attr_name: XAttrName, follow_symlinks: bool = True) -> bytes:
  raise OSError(errno.ENOTSUP, "xattr not supported")

def _setxattr_fallback(path: PathType, attr_name: XAttrName, attr_value: Buffer, flags: int = 0, follow_symlinks: bool = True) -> None:
  raise OSError(errno.ENOTSUP, "xattr not supported")

os_getxattr = getattr(os, 'getxattr', _getxattr_fallback)
os_setxattr = getattr(os, 'setxattr', _setxattr_fallback)

_cached_attributes: dict[tuple[str, str], bytes | None] = {}

def getxattr(path: str, attr_name: str) -> bytes | None:
  key = (path, attr_name)
  if key not in _cached_attributes:
    try:
      response = os_getxattr(path, attr_name, True)
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
  os_setxattr(path, attr_name, attr_value, 0, True)
