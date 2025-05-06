import ctypes
from collections.abc import Callable
from typing import Any, cast

# --- Type Definitions ---
GLenum = ctypes.c_uint
GLint = ctypes.c_int
GLsizei = ctypes.c_int
GLuint = ctypes.c_uint
GLfloat = ctypes.c_float
GLvoidp = ctypes.c_void_p

try:
  ctypes.pythonapi.dlsym.restype = ctypes.c_void_p
  ctypes.pythonapi.dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
  RTLD_DEFAULT = ctypes.c_void_p(0)
except AttributeError as err:
  raise ImportError("Cannot access ctypes.pythonapi.dlsym. Ensure you are on a POSIX-like system (Linux, macOS).") from err

_gl_func_cache: dict[str, Callable[..., Any]] = {}

def get_gl_function(name: str, restype: Any | None, argtypes: list) -> Callable[..., Any]:
  """
  Retrieves an OpenGL function pointer using dlsym after an OpenGL context
  is expected to be active. Caches the result.

  Raises AttributeError if the function cannot be found.
  """
  if name in _gl_func_cache:
    return _gl_func_cache[name]
  addr = ctypes.pythonapi.dlsym(RTLD_DEFAULT, name.encode('utf-8'))
  if not addr:
    raise AttributeError(f"OpenGL function '{name}' not found using dlsym. Ensure OpenGL context is active and function is supported.")
  proto = ctypes.CFUNCTYPE(restype, *argtypes)
  func = cast(Callable[..., Any], proto(addr))
  _gl_func_cache[name] = func
  return func


# --- OpenGL Function Wrappers ---


# void glPixelStorei(GLenum pname, GLint param)
def glPixelStorei(pname: GLenum, param: GLint) -> None:
  func = get_gl_function("glPixelStorei", None, [GLenum, GLint])
  func(pname, param)


# void glUniform1i(GLint location, GLint v0)
def glUniform1i(location: GLint, v0: GLint) -> None:
  func = get_gl_function("glUniform1i", None, [GLint, GLint])
  func(location, v0)


# void glTexImage2D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border,
#                   GLenum format, GLenum type, const GLvoid *pixels)
def glTexImage2D(
  target: GLenum,
  level: GLint,
  internalformat: GLint,  # Note: Often GLint, sometimes GLenum depending on GL version/spec interpretation
  width: GLsizei,
  height: GLsizei,
  border: GLint,
  _format: GLenum,
  _type: GLenum,
  pixels: ctypes.c_void_p | None,  # Can be None (or 0) to allocate texture without data
) -> None:
  func = get_gl_function("glTexImage2D", None, [GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, GLvoidp])
  func(target, level, internalformat, width, height, border, _format, _type, pixels)


# void glBindTexture(GLenum target, GLuint texture)
def glBindTexture(target: GLenum, texture: GLuint) -> None:
  func = get_gl_function("glBindTexture", None, [GLenum, GLuint])
  func(target, texture)


# void glActiveTexture(GLenum texture)
def glActiveTexture(texture: GLenum) -> None:
  func = get_gl_function("glActiveTexture", None, [GLenum])
  func(texture)


# GLenum glGetError(void)
def glGetError() -> GLenum:
  func = get_gl_function("glGetError", GLenum, [])
  return func()


def assert_gl_no_error() -> None:
  errno = glGetError()
  assert errno == GL_NO_ERROR, f"OpenGL error: {errno}"


# void glTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels)
def glTexSubImage2D(
  target: GLenum,
  level: GLint,
  xoffset: GLint,
  yoffset: GLint,
  width: GLsizei,
  height: GLsizei,
  _format: GLenum,
  _type: GLenum,
  pixels: ctypes.c_void_p | None,
) -> None:
  func = get_gl_function(
    "glTexSubImage2D",
    None,
    [GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, GLvoidp],
  )
  func(target, level, xoffset, yoffset, width, height, _format, _type, pixels)


# --- OpenGL Constants ---

GL_NO_ERROR = cast(GLenum, 0)

# Pixel storage parameters
GL_UNPACK_ROW_LENGTH = cast(GLenum, 0x0CF2)
GL_UNPACK_ALIGNMENT = cast(GLenum, 0x0CF5)

# Texture targets
GL_TEXTURE_2D = cast(GLenum, 0x0DE1)

# Texture parameters (pname for TexParameter*)
GL_TEXTURE_MIN_FILTER = cast(GLenum, 0x2801)
GL_TEXTURE_MAG_FILTER = cast(GLenum, 0x2800)
GL_TEXTURE_WRAP_S = cast(GLenum, 0x2802)
GL_TEXTURE_WRAP_T = cast(GLenum, 0x2803)

# Texture parameter values (param for TexParameter*)
GL_LINEAR = cast(GLint, 0x2601)  # Often used as GLint, but can be GLenum
GL_CLAMP_TO_EDGE = cast(GLint, 0x812F)  # Often used as GLint, but can be GLenum

# Internal formats (internalformat for TexImage*)
GL_R8 = cast(GLint, 0x8229)
GL_RG8 = cast(GLint, 0x822B)

# Pixel formats (format for TexImage*)
GL_RED = cast(GLenum, 0x1903)
GL_RG = cast(GLenum, 0x8227)

# Pixel types (type for TexImage*)
GL_UNSIGNED_BYTE = cast(GLenum, 0x1401)
