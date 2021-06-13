from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "selfdrive/common/params.cc":
  pass

cdef extern from "selfdrive/common/util.cc":
  pass

cdef extern from "selfdrive/common/params.h":
  cpdef enum ParamKeyType:
    PERSISTENT
    CLEAR_ON_MANAGER_START
    CLEAR_ON_PANDA_DISCONNECT
    CLEAR_ON_IGNITION_ON
    CLEAR_ON_IGNITION_OFF
    ALL

  cdef cppclass Params:
    Params(bool)
    Params(string)
    string get(string, bool) nogil
    bool getBool(string)
    int remove(string)
    int put(string, string)
    int putBool(string, bool)
    bool checkKey(string)
    void clearAll(ParamKeyType)
