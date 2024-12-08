# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

import os
from libcpp cimport bool
from libcpp.string cimport string

from .snpemodel cimport SNPEModel as cppSNPEModel
from selfdrive.modeld.models.commonmodel_pyx cimport CLContext
from selfdrive.modeld.runners.runmodel_pyx cimport RunModel
from selfdrive.modeld.runners.runmodel cimport RunModel as cppRunModel

os.environ['ADSP_LIBRARY_PATH'] = "/data/pythonpath/third_party/snpe/dsp/"

cdef class SNPEModel(RunModel):
  def __cinit__(self, string path, float[:] output, int runtime, bool use_tf8, CLContext context):
    self.model = <cppRunModel *> new cppSNPEModel(path, &output[0], len(output), runtime, use_tf8, context.context)
