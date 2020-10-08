# distutils: langauge = c++
import threading
import os
from libcpp cimport bool
from libcpp.string cimport string
from params_definition cimport Params as c_Params


class UnknownKeyName(Exception):
  pass

cdef class Params:
  cdef c_Params* p;

  def __cinit__(self):
    self.p = new c_Params(<bool>False)

  def __dealloc__(self):
    del self.p

  def get(self, key, block=False, encoding=None):
    if isinstance(key, str):
      key = key.encode('UTF-8')

    cdef string val = self.p.get(key)

    if val == b"":
      return None

    if encoding is not None:
      return val.decode(encoding)
    else:
      return val


def put_nonblocking(key, val, d=None):
  def f(key, val):
    params = Params(d)
    params.put(key, val)

  t = threading.Thread(target=f, args=(key, val))
  t.start()
  return t
