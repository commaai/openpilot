# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, nonecheck=False

import numpy as np
cimport numpy as np

np.import_array()

ctypedef np.uint8_t uint8_t

#---------------------------------------------------------------------------
# Extract YUV420 buffer and convert to RGB in a single pass
#---------------------------------------------------------------------------
cpdef np.ndarray[uint8_t, ndim=3, mode="c"] yuv420_to_rgb(object buf):
  '''
  Convert a packed YUV420 frame directly to RGB in one pass without
  intermediate full-size U/V planes.

  `buf.data` must be a contiguous 1D numpy.ndarray of dtype uint8.
  '''
  cdef int height    = buf.height
  cdef int width     = buf.width
  cdef int stride    = buf.stride
  cdef int uv_offset = buf.uv_offset

  # Allocate output RGB array (requires GIL)
  cdef np.ndarray[uint8_t, ndim=3, mode="c"] rgb = np.empty((height, width, 3), dtype=np.uint8)
  cdef uint8_t[:, :, :] out = rgb

  # Obtain memoryview of raw frame bytes
  cdef unsigned char[:] data = buf.data

  cdef int i, j
  cdef int idx_y, idx_uv
  cdef int Y, U, V, c_val, d_val, e_val, R, G, B

  # Main loop under nogil for speed
  with nogil:
    for i in range(height):
      for j in range(width):
        # compute indices
        idx_y  = i * stride + j
        idx_uv = uv_offset + (i >> 1) * stride + ((j >> 1) << 1)

        # load
        Y = data[idx_y]
        U = data[idx_uv]
        V = data[idx_uv + 1]

        # convert
        c_val = Y - 16
        d_val = U - 128
        e_val = V - 128

        R = (298 * c_val + 409 * e_val + 128) >> 8
        G = (298 * c_val - 100 * d_val - 208 * e_val + 128) >> 8
        B = (298 * c_val + 516 * d_val + 128) >> 8

        # store
        out[i, j, 0] = R
        out[i, j, 1] = G
        out[i, j, 2] = B

  return rgb
