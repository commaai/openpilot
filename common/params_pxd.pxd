from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "selfdrive/common/params.cc":
  pass

cdef extern from "selfdrive/common/util.c":
  pass

cdef extern from "selfdrive/common/params.h":
  cdef cppclass Params:
    Params(bool)
    Params(string)
    string get(string, bool) nogil
    int delete_db_value(string)
    int write_db_value(string, string)
