# distutils: language = c++

from msgq.visionipc.visionipc cimport cl_device_id, cl_context, cl_mem

cdef extern from "common/mat.h":
  cdef struct mat3:
    float v[9]

cdef extern from "common/clutil.h":
  cdef unsigned long CL_DEVICE_TYPE_DEFAULT
  cl_device_id cl_get_device_id(unsigned long)
  cl_context cl_create_context(cl_device_id)

cdef extern from "selfdrive/modeld/models/commonmodel.h":
  cppclass ModelFrame:
    int buf_size
    unsigned char * buffer_from_cl(cl_mem*, int);
    cl_mem * prepare(cl_mem, int, int, int, int, mat3)

  cppclass DrivingModelFrame:
    int buf_size
    DrivingModelFrame(cl_device_id, cl_context)

  cppclass MonitoringModelFrame:
    int buf_size
    MonitoringModelFrame(cl_device_id, cl_context)
