import builtins
import datetime
from enum import IntEnum, IntFlag
import json
from pathlib import Path

from cffi import FFI

from openpilot.common.swaglog import cloudlog


class ParamKeyFlag(IntFlag):
  PERSISTENT = 0x02
  CLEAR_ON_MANAGER_START = 0x04
  CLEAR_ON_ONROAD_TRANSITION = 0x08
  CLEAR_ON_OFFROAD_TRANSITION = 0x10
  DEVELOPMENT_ONLY = 0x40
  CLEAR_ON_IGNITION_ON = 0x80
  ALL = 0xFFFFFFFF


class ParamKeyType(IntEnum):
  STRING = 0
  BOOL = 1
  INT = 2
  FLOAT = 3
  TIME = 4
  JSON = 5
  BYTES = 6


ffi = FFI()
ffi.cdef("""
  typedef void ParamsHandle;
  ParamsHandle *params_create(const char *path);
  void params_destroy(ParamsHandle *p);
  const char *params_last_error(void);
  void params_clear_all(ParamsHandle *p, unsigned int flag);
  int params_check_key(ParamsHandle *p, const char *key);
  int params_get_key_type(ParamsHandle *p, const char *key);
  char *params_get_default(ParamsHandle *p, const char *key, size_t *size, int *present);
  char *params_get(ParamsHandle *p, const char *key, int block, size_t *size);
  int params_get_bool(ParamsHandle *p, const char *key, int block);
  int params_put(ParamsHandle *p, const char *key, const char *value, size_t size, int block);
  int params_put_bool(ParamsHandle *p, const char *key, int value, int block);
  int params_remove(ParamsHandle *p, const char *key);
  char *params_get_path(ParamsHandle *p, const char *key, size_t *size);
  size_t params_keys_size(ParamsHandle *p);
  char *params_key_at(ParamsHandle *p, size_t index, size_t *size);
  void params_free(void *value);
""")

_suffix = ".dylib" if __import__("sys").platform == "darwin" else ".so"
lib = ffi.dlopen(str(Path(__file__).with_name(f"libparams_cffi{_suffix}")))

PYTHON_2_CPP = {
  (str, ParamKeyType.STRING): lambda v: v,
  (builtins.bool, ParamKeyType.BOOL): lambda v: "1" if v else "0",
  (int, ParamKeyType.INT): str,
  (float, ParamKeyType.FLOAT): str,
  (datetime.datetime, ParamKeyType.TIME): lambda v: v.isoformat(),
  (dict, ParamKeyType.JSON): json.dumps,
  (list, ParamKeyType.JSON): json.dumps,
  (bytes, ParamKeyType.BYTES): lambda v: v,
}
CPP_2_PYTHON = {
  ParamKeyType.STRING: lambda v: v.decode("utf-8"),
  ParamKeyType.BOOL: lambda v: v == b"1",
  ParamKeyType.INT: int,
  ParamKeyType.FLOAT: float,
  ParamKeyType.TIME: lambda v: datetime.datetime.fromisoformat(v.decode("utf-8")),
  ParamKeyType.JSON: json.loads,
  ParamKeyType.BYTES: lambda v: v,
}


def ensure_bytes(v):
  return v.encode() if isinstance(v, str) else v


def _take_string(value, size):
  if value == ffi.NULL:
    return None
  try:
    return bytes(ffi.buffer(value, size))
  finally:
    lib.params_free(value)


class UnknownKeyName(Exception):
  pass


class Params:
  def __init__(self, d=""):
    self.p = lib.params_create(ensure_bytes(d))
    if self.p == ffi.NULL:
      raise RuntimeError(ffi.string(lib.params_last_error()).decode())
    self.p = ffi.gc(self.p, lib.params_destroy)
    self.d = d

  def __reduce__(self):
    return (type(self), (self.d,))

  def clear_all(self, tx_flag=ParamKeyFlag.ALL):
    lib.params_clear_all(self.p, int(tx_flag))

  def check_key(self, key):
    key = ensure_bytes(key)
    if not lib.params_check_key(self.p, key):
      raise UnknownKeyName(key)
    return key

  def python2cpp(self, proposed_type, expected_type, value, key):
    cast = PYTHON_2_CPP.get((proposed_type, expected_type))
    if cast:
      return cast(value)
    raise TypeError(f"Type mismatch while writing param {key}: {proposed_type=} {expected_type=} {value=}")

  def _cpp2python(self, t, value, default, key):
    if value is None:
      return None
    try:
      return CPP_2_PYTHON[t](value)
    except (KeyError, TypeError, ValueError):
      cloudlog.warning(f"Failed to cast param {key} with {value=} from type {t=}")
      return self._cpp2python(t, default, None, key)

  def _default(self, key):
    size, present = ffi.new("size_t *"), ffi.new("int *")
    value = lib.params_get_default(self.p, key, size, present)
    return _take_string(value, size[0]) if present[0] else None

  def get(self, key, block=False, return_default=False):
    k = self.check_key(key)
    t = self.get_type(k)
    default = self._default(k) if return_default else None
    size = ffi.new("size_t *")
    value = _take_string(lib.params_get(self.p, k, block, size), size[0])
    if value == b"":
      if block:
        raise KeyboardInterrupt
      return self._cpp2python(t, default, None, key)
    return self._cpp2python(t, value, default, key)

  def get_bool(self, key, block=False):
    return bool(lib.params_get_bool(self.p, self.check_key(key), block))

  def _put_cast(self, key, dat):
    return ensure_bytes(self.python2cpp(type(dat), self.get_type(key), dat, key))

  def put(self, key, dat, block=False):
    """Write a parameter. block=True waits until it is persisted to disk."""
    k = self.check_key(key)
    value = self._put_cast(k, dat)
    lib.params_put(self.p, k, value, len(value), block)

  def put_bool(self, key, val, block=False):
    lib.params_put_bool(self.p, self.check_key(key), val, block)

  def remove(self, key):
    lib.params_remove(self.p, self.check_key(key))

  def get_param_path(self, key=""):
    size = ffi.new("size_t *")
    return _take_string(lib.params_get_path(self.p, ensure_bytes(key), size), size[0]).decode()

  def get_type(self, key):
    return ParamKeyType(lib.params_get_key_type(self.p, self.check_key(key)))

  def all_keys(self):
    keys = []
    for i in range(lib.params_keys_size(self.p)):
      size = ffi.new("size_t *")
      keys.append(_take_string(lib.params_key_at(self.p, i, size), size[0]))
    return keys

  def get_default_value(self, key):
    k = self.check_key(key)
    return self._cpp2python(self.get_type(k), self._default(k), None, key)

  def cpp2python(self, key, value):
    return self._cpp2python(self.get_type(key), value, None, key)


if __name__ == "__main__":
  import sys

  params = Params()
  key = sys.argv[1]
  params.check_key(key)
  if len(sys.argv) == 3:
    val = sys.argv[2]
    print(f"SET: {key} = {val}")
    params.put(key, val, block=True)
  elif len(sys.argv) == 2:
    print(f"GET: {key} = {params.get(key)}")
