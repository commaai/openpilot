from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "../selfdrive/common/params_helper.cc":
  pass

# Declare the class with cdef
cdef extern from "../selfdrive/common/params_helper.h" namespace "params":
  cdef cppclass Params:
    string db;
    Params(string) except +
    void clear_all() except+
    void manager_start() except+
    void panda_disconnect() except+
    void _delete(string) except+
    string get(string, bool) nogil except+
    void put(string, string) nogil except+
