# distutils: language = c++

from .runmodel cimport RunModel as cppRunModel

cdef class RunModel:
  cdef cppRunModel * model
