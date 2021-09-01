import os
import subprocess
from cffi import FFI
from common.basedir import BASEDIR

# Initialize visiontest. Ignore output.
_visiond_dir = os.path.dirname(os.path.abspath(__file__))
_libvisiontest = "libvisiontest.so"
try:  # because this crashes sometimes when running pipeline
  subprocess.check_output(["make", "-C", _visiond_dir, "-f",
                           os.path.join(_visiond_dir, "visiontest.mk"),
                           _libvisiontest])
except Exception:
  pass


class VisionTest():
  """A version of the vision model that can be run on a desktop.

     WARNING: This class is not thread safe. VisionTest objects cannot be
              created or used on multiple threads simultaneously.
  """

  ffi = FFI()
  ffi.cdef("""
  typedef unsigned char uint8_t;

  struct VisionTest;
  typedef struct VisionTest VisionTest;

  VisionTest* visiontest_create(int temporal_model, int disable_model,
                                int input_width, int input_height,
                                int model_input_width, int model_input_height);
  void visiontest_destroy(VisionTest* visiontest);

  void visiontest_transform(VisionTest* vt, const uint8_t* yuv_data,
                            uint8_t* out_y, uint8_t* out_u, uint8_t* out_v,
                            const float* transform);
  """)

  clib = ffi.dlopen(os.path.join(_visiond_dir, _libvisiontest))

  def __init__(self, input_size, model_input_size, model):
    """Create a wrapper around visiond for off-device python code.

       Inputs:
        input_size: The size of YUV images passed to transform.
        model_input_size: The size of YUV images passed to the model.
        model: The name of the model to use. "temporal", "yuv", or None to disable the
               model (used to disable OpenCL).
    """
    self._input_size = input_size
    self._model_input_size = model_input_size

    if model is None:
      disable_model = 1
      temporal_model = 0
    elif model == "yuv":
      disable_model = 0
      temporal_model = 0
    elif model == "temporal":
      disable_model = 0
      temporal_model = 1
    else:
      raise ValueError("Bad model name: {}".format(model))

    prevdir = os.getcwd()
    os.chdir(_visiond_dir)  # tmp hack to find kernels
    os.environ['BASEDIR'] = BASEDIR
    self._visiontest_c = self.clib.visiontest_create(
      temporal_model, disable_model, self._input_size[0], self._input_size[1],
      self._model_input_size[0], self._model_input_size[1])
    os.chdir(prevdir)

  @property
  def input_size(self):
    return self._input_size

  @property
  def model_input_size(self):
    return self._model_input_size

  def transform(self, yuv_data, transform):
    y_len = self.model_input_size[0] * self.model_input_size[1]
    t_y_ptr = bytearray(y_len)
    t_u_ptr = bytearray(y_len // 4)
    t_v_ptr = bytearray(y_len // 4)

    self.transform_output_buffer(yuv_data, t_y_ptr, t_u_ptr, t_v_ptr,
                                 transform)

    return t_y_ptr, t_u_ptr, t_v_ptr

  def transform_contiguous(self, yuv_data, transform):
    y_ol = self.model_input_size[0] * self.model_input_size[1]
    uv_ol = y_ol // 4
    result = bytearray(y_ol * 3 // 2)
    result_view = memoryview(result)
    t_y_ptr = result_view[:y_ol]
    t_u_ptr = result_view[y_ol:y_ol + uv_ol]
    t_v_ptr = result_view[y_ol + uv_ol:]

    self.transform_output_buffer(yuv_data, t_y_ptr, t_u_ptr, t_v_ptr,
                                 transform)
    return result

  def transform_output_buffer(self, yuv_data, y_out, u_out, v_out,
                              transform):
    assert len(yuv_data) == self.input_size[0] * self.input_size[1] * 3 / 2

    cast = self.ffi.cast
    from_buffer = self.ffi.from_buffer
    yuv_ptr = cast("unsigned char*", from_buffer(yuv_data))
    transform_ptr = self.ffi.new("float[]", transform)

    y_out_ptr = cast("unsigned char*", from_buffer(y_out))
    u_out_ptr = cast("unsigned char*", from_buffer(u_out))
    v_out_ptr = cast("unsigned char*", from_buffer(v_out))

    self.clib.visiontest_transform(self._visiontest_c, yuv_ptr, y_out_ptr,
                                   u_out_ptr, v_out_ptr, transform_ptr)

  def close(self):
    self.clib.visiontest_destroy(self._visiontest_c)
    self._visiontest_c = None

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()


if __name__ == "__main__":
  VisionTest((560, 304), (320, 160), "temporal")
