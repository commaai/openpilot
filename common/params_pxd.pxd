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
    Params(bool) nogil
    Params(string) nogil
    string get(string, bool) nogil
    bool getBool(string) nogil
    int remove(string) nogil
    int put(string, string) nogil
    int putBool(string, bool) nogil
    bool checkKey(string) nogil
    void clearAll(ParamKeyType)
