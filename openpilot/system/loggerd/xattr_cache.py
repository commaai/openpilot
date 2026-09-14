import ctypes
import errno
import os
import sys


if sys.platform == "darwin":
  _libc = ctypes.CDLL(None, use_errno=True)
  _libc.getxattr.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_int]
  _libc.getxattr.restype = ctypes.c_ssize_t
  _libc.setxattr.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_int]
  _libc.setxattr.restype = ctypes.c_int


def _raise_os_error(path: str) -> None:
  error = ctypes.get_errno()
  raise OSError(error, os.strerror(error), path)


def _getxattr(path: str, attr_name: str) -> bytes:
  if sys.platform != "darwin":
    return os.getxattr(path, attr_name)

  encoded_path = os.fsencode(path)
  encoded_attr_name = os.fsencode(attr_name)
  while True:
    size = _libc.getxattr(encoded_path, encoded_attr_name, None, 0, 0, 0)
    if size == -1:
      _raise_os_error(path)
    if size == 0:
      return b""

    value = ctypes.create_string_buffer(size)
    result = _libc.getxattr(encoded_path, encoded_attr_name, value, size, 0, 0)
    if result != -1:
      return value.raw[:result]
    if ctypes.get_errno() != errno.ERANGE:
      _raise_os_error(path)


def _setxattr(path: str, attr_name: str, attr_value: bytes) -> None:
  if sys.platform != "darwin":
    os.setxattr(path, attr_name, attr_value)
    return

  encoded_path = os.fsencode(path)
  encoded_attr_name = os.fsencode(attr_name)
  value = ctypes.create_string_buffer(attr_value)
  if _libc.setxattr(encoded_path, encoded_attr_name, value, len(attr_value), 0, 0) == -1:
    _raise_os_error(path)

_cached_attributes: dict[tuple, bytes | None] = {}

def getxattr(path: str, attr_name: str) -> bytes | None:
  key = (path, attr_name)
  if key not in _cached_attributes:
    try:
      response = _getxattr(path, attr_name)
    except OSError as e:
      # ENODATA (Linux) or ENOATTR (macOS) means attribute hasn't been set
      if e.errno == errno.ENODATA or (hasattr(errno, 'ENOATTR') and e.errno == errno.ENOATTR):
        response = None
      else:
        raise
    _cached_attributes[key] = response
  return _cached_attributes[key]

def setxattr(path: str, attr_name: str, attr_value: bytes) -> None:
  _cached_attributes.pop((path, attr_name), None)
  _setxattr(path, attr_name, attr_value)
