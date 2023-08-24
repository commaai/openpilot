# distutils: language = c++
#cython: language_level=3

from .runmodel cimport RunModel as cppRunModel

cdef class RunModel:
  cdef cppRunModel * model
