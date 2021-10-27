from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "selfdrive/ui/replay/framereader.cc":
  pass

cdef extern from "selfdrive/ui/replay/framereader.h":
  cdef cppclass FrameReader:
    FrameReader()
    bool load(string)
    int getRGBSize()
    int getYUVSize()
    int frame_count()
    void *get(int)
    int width
    int height

