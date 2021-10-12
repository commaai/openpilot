from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "selfdrive/ui/replay/logreader.cc":
  pass

cdef extern from "selfdrive/ui/replay/logreader.h":
  cdef cppclass LogReader:
    LogReader()
    bool load(string, bool)
