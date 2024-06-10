# distutils: language = c++
# cython: language_level = 3
import sys

from libcpp.string cimport string


cdef extern from "common/proc_title.cc":
  void setProcTitle(char*)
  cdef string getProcTitle()
  void _init(int, char*)


cdef init():
  argv = sys.orig_argv
  argc = len(argv)
  _init(argc, argv[0].encode())


def set_proc_title(title):
  if sys.platform not in ('linux'):
    return
  setProcTitle(title.encode())


def get_proc_title():
  if sys.platform not in ('linux'):
    return 0
  return getProcTitle().decode()

init()
