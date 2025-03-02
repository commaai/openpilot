# distutils: language = c++

from msgq.visionipc.visionipc cimport cl_device_id, cl_context, cl_mem

cdef extern from "common/mat.h":
  cdef struct mat3:
    float v[9]

cdef extern from "common/clutil.h":
  cdef unsigned long CL_DEVICE_TYPE_DEFAULT
  cl_device_id cl_get_device_id(unsigned long)
  cl_context cl_create_context(cl_device_id)
  void cl_release_context(cl_context)

cdef extern from "sunnypilot/modeld_v2/models/commonmodel.h":
  cppclass ModelFrame:
    int buf_size
    unsigned char * buffer_from_cl(cl_mem*, int);
    cl_mem * prepare(cl_mem, int, int, int, int, mat3)

  cppclass DrivingModelFrame:
    int buf_size
    DrivingModelFrame(cl_device_id, cl_context, unsigned char)

  cppclass MonitoringModelFrame:
    int buf_size
    MonitoringModelFrame(cl_device_id, cl_context)
