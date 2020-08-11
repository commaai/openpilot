# distutils: langauge = c++
from common.basedir import PARAMS
import threading
from libcpp cimport bool
from libcpp.string cimport string
from params_definition cimport Params as c_Params

class UnknownKeyName(Exception):
  pass

cdef class Params:
  cdef c_Params* p;

  def __cinit__(self, d=PARAMS):
    try:
      d = d.encode('UTF-8')
    except (UnicodeDecodeError, AttributeError):
      pass
    self.p = new c_Params(d)

  def clear_all(self):
    self.p.clear_all()

  def manager_start(self):
    self.p.manager_start()

  def panda_disconnect(self):
    self.p.panda_disconnect()

  def delete(self, key):
    try:
      key = key.encode('UTF-8')
    except (UnicodeDecodeError, AttributeError):
      pass
    self.p._delete(key)


  # Parameterized blocking/non-blocking get function.
  # The blocking variant releases the GIL.
  def get(self, key, block=False, encoding=None):
    try:
      key = key.encode('UTF-8')
    except (UnicodeDecodeError, AttributeError):
      pass

    cdef string val
    cdef string k = key
    cdef bool b = block
    try:
      if block:
        with nogil:
          val = self.p.get(k, b)
      else:
        val = self.p.get(key, block)
    except (RuntimeError):
      raise UnknownKeyName(key)

    ret = val
    if (ret == b""):
      ret = None

    if ret is not None and encoding is not None:
      ret = ret.decode(encoding)
    return ret

  # This put function can block, so there is a non-blocking variant
  # which can be used in time-sensitive code. Writing Params, should
  # be avoided. This function releases the GIL.
  def put(self, key, value):
    try:
      key = key.encode('UTF-8')
      value = value.encode('UTF-8')
    except (UnicodeDecodeError, AttributeError):
      pass
    cdef string k = key;
    cdef string v = value;
    try:
        with nogil:
          self.p.put(k, v)
    except (RuntimeError):
      raise UnknownKeyName(key)
  
  def __dealloc__(self):
    del self.p

def put_nonblocking(key, val):
  def f(key, val):
    params = Params()
    params.put(key, val)

  t = threading.Thread(target=f, args=(key, val))
  t.start()
  return t
