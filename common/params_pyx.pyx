# distutils: language = c++
# cython: language_level = 3
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "common/params.h" nogil:
  cpdef enum ParamKeyType:
    PERSISTENT
    CLEAR_ON_MANAGER_START
    CLEAR_ON_ONROAD_TRANSITION
    CLEAR_ON_OFFROAD_TRANSITION
    DEVELOPMENT_ONLY
    ALL

  cdef cppclass c_Params "Params":
    c_Params(string) except +
    string get(string, bool)
    bool getBool(string, bool)
    int remove(string)
    int put(string, string)
    void putNonBlocking(string, string)
    void putBoolNonBlocking(string, bool)
    int putBool(string, bool)
    bool checkKey(string)
    string getParamPath(string)
    void clearAll(ParamKeyType)
    vector[string] allKeys()


def ensure_bytes(v):
  return v.encode() if isinstance(v, str) else v

class UnknownKeyName(Exception):
  pass

cdef class Params:
  cdef c_Params* p
  cdef str d

  def __cinit__(self, d=""):
    cdef string path = <string>d.encode()
    with nogil:
      self.p = new c_Params(path)
    self.d = d

  def __reduce__(self):
    return (type(self), (self.d,))

  def __dealloc__(self):
    with nogil:
      del self.p

  def clear_all(self, ParamKeyType key_type=ParamKeyType.ALL):
    with nogil:
      self.p.clearAll(key_type)

  def check_key(self, key):
    cdef string cpp_key = ensure_bytes(key)
    cdef bool exists
    with nogil:
      exists = self.p.checkKey(cpp_key)
    if not exists:
      raise UnknownKeyName(key)
    return cpp_key

  def get(self, key, bool block=False, encoding=None):
    cdef string k = self.check_key(key)
    cdef string val
    with nogil:
      val = self.p.get(k, block)

    if val == b"":
      if block:
        # If we got no value while running in blocked mode
        # it means we got an interrupt while waiting
        raise KeyboardInterrupt
      else:
        return None

    return val if encoding is None else val.decode(encoding)

  def get_bool(self, key, bool block=False):
    cdef string k = self.check_key(key)
    cdef bool r
    with nogil:
      r = self.p.getBool(k, block)
    return r

  def put(self, key, dat):
    """
    Warning: This function blocks until the param is written to disk!
    In very rare cases this can take over a second, and your code will hang.
    Use the put_nonblocking, put_bool_nonblocking in time sensitive code, but
    in general try to avoid writing params as much as possible.
    """
    cdef string k = self.check_key(key)
    cdef string dat_bytes = ensure_bytes(dat)
    with nogil:
      self.p.put(k, dat_bytes)

  def put_bool(self, key, bool val):
    cdef string k = self.check_key(key)
    with nogil:
      self.p.putBool(k, val)

  def put_nonblocking(self, key, dat):
    cdef string k = self.check_key(key)
    cdef string dat_bytes = ensure_bytes(dat)
    with nogil:
      self.p.putNonBlocking(k, dat_bytes)

  def put_bool_nonblocking(self, key, bool val):
    cdef string k = self.check_key(key)
    with nogil:
      self.p.putBoolNonBlocking(k, val)

  def remove(self, key):
    cdef string k = self.check_key(key)
    with nogil:
      self.p.remove(k)

  def get_param_path(self, key=""):
    cdef string key_bytes = ensure_bytes(key)
    cdef string result
    with nogil:
      result = self.p.getParamPath(key_bytes)
    return result.decode('utf-8')

  def all_keys(self):
    cdef vector[string] keys
    with nogil:
      keys = self.p.allKeys()
    return [key.decode('utf-8') for key in keys]
