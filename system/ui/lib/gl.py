import ctypes.util
from collections.abc import Callable
from typing import cast

# Define OpenGL types
GLenum = ctypes.c_uint
GLint = ctypes.c_int

# OpenGL pixel store parameters
GL_UNPACK_ALIGNMENT = cast(GLenum, 0x0CF5)
GL_UNPACK_ROW_LENGTH = cast(GLenum, 0x0CF2)

# Load OpenGL library
gl_lib_name = ctypes.util.find_library('GL')
if gl_lib_name is None:
  raise ImportError("Could not find OpenGL library")
gl = ctypes.CDLL(gl_lib_name)

# Define glPixelStorei function pointer with type hints
glPixelStorei: Callable[[GLenum, GLint], None]

try:
  glPixelStorei = gl.glPixelStorei
  glPixelStorei.argtypes = [GLenum, GLint]
  glPixelStorei.restype = None
except AttributeError:
  print("glPixelStorei function not found in OpenGL library")
