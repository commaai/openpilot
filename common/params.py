import os
import ctypes
from ctypes import c_void_p, c_char_p, c_int, c_bool, c_size_t, POINTER
from enum import IntEnum, IntFlag

# Load library
def get_libpath():
  # Try strict path relative to this file
  path = os.path.dirname(os.path.realpath(__file__))
  libname = "libparams_c.so"
  return os.path.join(path, libname)

try:
  _libparams = ctypes.CDLL(get_libpath())
  _libc = ctypes.CDLL("libc.so.6")
except OSError:
  # Fallback for testing or if not built yet
  _libparams = None

class ParamKeyType(IntEnum):
  STRING = 0
  BOOL = 1
  INT = 2
  FLOAT = 3
  TIME = 4
  JSON = 5
  BYTES = 6

class ParamKeyFlag(IntFlag):
  PERSISTENT = 0x02
  CLEAR_ON_MANAGER_START = 0x04
  CLEAR_ON_ONROAD_TRANSITION = 0x08
  CLEAR_ON_OFFROAD_TRANSITION = 0x10
  DONT_LOG = 0x20
  DEVELOPMENT_ONLY = 0x40
  CLEAR_ON_IGNITION_ON = 0x80
  ALL = 0xFFFFFFFF

class UnknownKeyName(Exception):
  pass

if _libparams:
  # Setup function signatures
  _libparams.params_create.argtypes = [c_char_p]
  _libparams.params_create.restype = c_void_p

  _libparams.params_destroy.argtypes = [c_void_p]
  _libparams.params_destroy.restype = None

  _libparams.params_check_key.argtypes = [c_void_p, c_char_p]
  _libparams.params_check_key.restype = c_bool

  _libparams.params_get_key_flag.argtypes = [c_void_p, c_char_p]
  _libparams.params_get_key_flag.restype = c_int

  _libparams.params_get_key_type.argtypes = [c_void_p, c_char_p]
  _libparams.params_get_key_type.restype = c_int

  _libparams.params_get.argtypes = [c_void_p, c_char_p, c_bool, POINTER(c_size_t)]
  _libparams.params_get.restype = c_void_p  # returns malloc'd buffer

  _libparams.params_put.argtypes = [c_void_p, c_char_p, c_char_p, c_size_t]
  _libparams.params_put.restype = c_int

  _libparams.params_put_bool.argtypes = [c_void_p, c_char_p, c_bool]
  _libparams.params_put_bool.restype = c_int

  _libparams.params_put_nonblocking.argtypes = [c_void_p, c_char_p, c_char_p, c_size_t]
  _libparams.params_put_nonblocking.restype = None

  _libparams.params_put_bool_nonblocking.argtypes = [c_void_p, c_char_p, c_bool]
  _libparams.params_put_bool_nonblocking.restype = None

  _libparams.params_get_param_path.argtypes = [c_void_p, c_char_p]
  _libparams.params_get_param_path.restype = c_void_p # char*

  _libparams.params_all_keys.argtypes = [c_void_p, POINTER(c_size_t)]
  _libparams.params_all_keys.restype = POINTER(c_char_p)

  _libparams.params_free_str_array.argtypes = [POINTER(c_char_p), c_size_t]
  _libparams.params_free_str_array.restype = None

  _libparams.params_get_default_value.argtypes = [c_void_p, c_char_p]
  _libparams.params_get_default_value.restype = c_void_p # char*

  _libparams.params_remove.argtypes = [c_void_p, c_char_p]
  _libparams.params_remove.restype = c_int

  _libparams.params_clear_all.argtypes = [c_void_p, c_int]
  _libparams.params_clear_all.restype = None

class Params:
  def __init__(self, path=None):
    if _libparams is None:
      raise RuntimeError("libparams_c.so not found. Run scons to build it.")

    self.path = path.encode('utf-8') if path else None
    self.ptr = _libparams.params_create(self.path)

  def __del__(self):
    if getattr(self, 'ptr', None):
      _libparams.params_destroy(self.ptr)

  def check_key(self, key):
    return _libparams.params_check_key(self.ptr, key.encode('utf-8'))

  def get_key_flag(self, key):
    if not self.check_key(key):
      raise UnknownKeyName(key)
    return _libparams.params_get_key_flag(self.ptr, key.encode('utf-8'))

  def get_key_type(self, key):
    if not self.check_key(key):
      raise UnknownKeyName(key)
    return _libparams.params_get_key_type(self.ptr, key.encode('utf-8'))

  def get(self, key, block=False, encoding='utf_8', return_default=False):
    if not self.check_key(key):
      raise UnknownKeyName(key)

    key_bytes = key.encode('utf-8')
    size = c_size_t(0)
    ret_ptr = _libparams.params_get(self.ptr, key_bytes, block, ctypes.byref(size))

    data = None
    if ret_ptr:
      try:
        data = ctypes.string_at(ret_ptr, size.value)
      finally:
        _libc.free(ret_ptr)

    if data is None and return_default:
       ptr = _libparams.params_get_default_value(self.ptr, key_bytes)
       if ptr:
         try:
           data = ctypes.string_at(ptr)
         finally:
           _libc.free(ptr)

    if data is None:
      return None

    # Determine type and convert
    try:
      key_type = self.get_key_type(key)
      if key_type == ParamKeyType.STRING:
        return data.decode(encoding)
      elif key_type == ParamKeyType.BOOL:
        return data == b"1"
      elif key_type == ParamKeyType.INT:
        return int(data)
      elif key_type == ParamKeyType.FLOAT:
        return float(data)
      elif key_type == ParamKeyType.TIME:
        import datetime
        return datetime.datetime.fromisoformat(data.decode(encoding))
      elif key_type == ParamKeyType.JSON:
        import json
        return json.loads(data)
      elif key_type == ParamKeyType.BYTES:
        return data
      else:
        return data
    except Exception:
        return self.get_default_value(key)

  def get_bool(self, key, block=False):
    if not self.check_key(key):
      raise UnknownKeyName(key)

    val = self.get(key, block)
    if val is None:
      return False
    if isinstance(val, bool):
      return val
    return val == b"1" or val == "1"

  def put(self, key, val):
    if not self.check_key(key):
      raise UnknownKeyName(key)

    key_bytes = key.encode('utf-8')
    if isinstance(val, str):
      val_bytes = val.encode('utf-8')
    elif isinstance(val, bool):
      return self.put_bool(key, val)
    elif isinstance(val, (int, float)):
      val_bytes = str(val).encode('utf-8')
    elif isinstance(val, dict):
      import json
      val_bytes = json.dumps(val).encode('utf-8')
    elif hasattr(val, "isoformat"): # datetime
      val_bytes = val.isoformat().encode('utf-8')
    elif isinstance(val, bytes):
      val_bytes = val
    else:
      val_bytes = val # Try direct if ctypes accepts it (or conversion fails later)

    return _libparams.params_put(self.ptr, key_bytes, val_bytes, len(val_bytes))

  def put_bool(self, key, val):
    if not self.check_key(key):
      raise UnknownKeyName(key)
    return _libparams.params_put_bool(self.ptr, key.encode('utf-8'), val)

  def put_nonblocking(self, key, val):
    if not self.check_key(key):
      raise UnknownKeyName(key)

    key_bytes = key.encode('utf-8')
    if isinstance(val, str):
      val_bytes = val.encode('utf-8')
    elif isinstance(val, bool):
      self.put_bool_nonblocking(key, val)
      return
    elif isinstance(val, (int, float)):
      val_bytes = str(val).encode('utf-8')
    elif isinstance(val, dict):
      import json
      val_bytes = json.dumps(val).encode('utf-8')
    elif hasattr(val, "isoformat"):
      val_bytes = val.isoformat().encode('utf-8')
    elif isinstance(val, bytes):
      val_bytes = val
    else:
      val_bytes = val

    _libparams.params_put_nonblocking(self.ptr, key_bytes, val_bytes, len(val_bytes))

  def put_bool_nonblocking(self, key, val):
    if not self.check_key(key):
      raise UnknownKeyName(key)
    _libparams.params_put_bool_nonblocking(self.ptr, key.encode('utf-8'), val)

  def remove(self, key):
    if not self.check_key(key):
      raise UnknownKeyName(key)
    return _libparams.params_remove(self.ptr, key.encode('utf-8'))

  def clear_all(self, flag=ParamKeyFlag.ALL):
    _libparams.params_clear_all(self.ptr, int(flag))

  def get_param_path(self, key=None):
    key_bytes = key.encode('utf-8') if key else None
    ptr = _libparams.params_get_param_path(self.ptr, key_bytes)
    try:
      return ctypes.string_at(ptr).decode('utf-8')
    finally:
      _libc.free(ptr)

  def all_keys(self):
    size = c_size_t(0)
    keys_ptr = _libparams.params_all_keys(self.ptr, ctypes.byref(size))
    if not keys_ptr:
      return []

    try:
      keys = []
      for i in range(size.value):
        k = keys_ptr[i]
        keys.append(k) # Return bytes, as k is C-string bytes
      return keys
    finally:
      _libparams.params_free_str_array(keys_ptr, size)

  def get_default_value(self, key):
    if not self.check_key(key):
      raise UnknownKeyName(key)
    ptr = _libparams.params_get_default_value(self.ptr, key.encode('utf-8'))
    if not ptr:
      return None
    try:
      data = ctypes.string_at(ptr)

      key_type = self.get_key_type(key)
      if key_type == ParamKeyType.STRING:
          return data.decode('utf-8')
      elif key_type == ParamKeyType.BOOL:
          return data == b"1"
      elif key_type == ParamKeyType.INT:
          return int(data)
      elif key_type == ParamKeyType.FLOAT:
          return float(data)
      elif key_type == ParamKeyType.TIME:
          import datetime
          return datetime.datetime.fromisoformat(data.decode('utf-8'))
      elif key_type == ParamKeyType.JSON:
          import json
          return json.loads(data)
      else:
          return data
    finally:
      _libc.free(ptr)

if __name__ == "__main__":
  import sys
  params = Params()
  if len(sys.argv) > 1:
      key = sys.argv[1]
      assert params.check_key(key), f"unknown param: {key}"
      if len(sys.argv) == 3:
          val = sys.argv[2]
          print(f"SET: {key} = {val}")
          params.put(key, val)
      elif len(sys.argv) == 2:
          print(f"GET: {key} = {params.get(key)}")
