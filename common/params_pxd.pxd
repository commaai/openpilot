from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "selfdrive/common/params.cc":
  pass

cdef extern from "selfdrive/common/util.cc":
  pass

cdef extern from "selfdrive/common/params.h":
  cdef cppclass Params:
    Params(bool)
    Params(string)
    string get(string, bool) nogil
    int remove(string)
    int put(string, string)
