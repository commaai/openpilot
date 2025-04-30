import ctypes
import sys
from collections.abc import Callable
from typing import Any, cast

# --- OpenGL Type Definitions ---
# Basic types
GLenum = ctypes.c_uint
GLboolean = ctypes.c_ubyte  # unsigned char
GLbitfield = ctypes.c_uint
GLbyte = ctypes.c_byte  # signed char
GLshort = ctypes.c_short
GLint = ctypes.c_int
GLsizei = ctypes.c_int
GLubyte = ctypes.c_ubyte
GLushort = ctypes.c_ushort
GLuint = ctypes.c_uint
GLfloat = ctypes.c_float
GLclampf = ctypes.c_float
GLdouble = ctypes.c_double
GLclampd = ctypes.c_double
GLvoidp = ctypes.c_void_p
GLintptr = ctypes.c_ssize_t  # Corresponds to ptrdiff_t / ssize_t
GLsizeiptr = ctypes.c_size_t  # Corresponds to size_t

if sys.platform.startswith('linux') or sys.platform == 'darwin':
  try:
    # Access dlsym function from python's C API
    ctypes.pythonapi.dlsym.restype = ctypes.c_void_p
    ctypes.pythonapi.dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    RTLD_DEFAULT = ctypes.c_void_p(0)  # Handle for searching all loaded symbols
  except AttributeError as err:
    raise ImportError("Cannot access ctypes.pythonapi.dlsym. Ensure you are on a POSIX-like system (Linux, macOS).") from err
else:
  raise ImportError(f"Platform '{sys.platform}' not supported by this gl.py version (requires dlsym).")

# --- Generic Function Loader ---
_gl_func_cache: dict[str, Callable[..., Any]] = {}  # Cache for loaded functions


def get_gl_function(
  name: str,
  restype: Any | None,  # e.g., None, GLenum, POINTER(GLuint)
  argtypes: list,  # e.g., [GLenum, GLint], [GLsizei, POINTER(GLuint)]
) -> Callable[..., Any]:
  """
  Retrieves an OpenGL function pointer using dlsym after an OpenGL context
  is expected to be active. Caches the result.

  Raises AttributeError if the function cannot be found.
  """
  if name in _gl_func_cache:
    return _gl_func_cache[name]

  # Convert name to bytes for dlsym
  name_bytes = name.encode('utf-8')

  # Use dlsym to find the function address in the current process space
  # This relies on Raylib/GLFW having already loaded the necessary GL functions.
  func_ptr_addr = ctypes.pythonapi.dlsym(RTLD_DEFAULT, name_bytes)

  if not func_ptr_addr:
    # Function not found. This might happen if the GL context is not
    # yet created, the function is not supported by the driver, or
    # it's an extension function that needs specific loading.
    raise AttributeError(f"OpenGL function '{name}' not found using dlsym. Ensure OpenGL context is active and function is supported.")

  # Create the ctypes function pointer object with the correct signature
  func_proto = ctypes.CFUNCTYPE(restype, *argtypes)
  func_ptr = cast(Callable[..., Any], func_proto(func_ptr_addr))  # Cast to generic Callable

  # Cache and return
  _gl_func_cache[name] = func_ptr
  # print(f"Loaded OpenGL function: {name}") # Optional debug log
  return func_ptr


# --- OpenGL Function Definitions ---


# glPixelStorei(GLenum pname, GLint param)
def glPixelStorei(pname: GLenum, param: GLint) -> None:
  func = get_gl_function("glPixelStorei", None, [GLenum, GLint])
  func(pname, param)


# glGenTextures(GLsizei n, GLuint *textures)
def glGenTextures(n: GLsizei) -> list[GLuint]:
  """Generates texture names."""
  func = get_gl_function("glGenTextures", None, [GLsizei, ctypes.POINTER(GLuint)])
  texture_ids = (GLuint * n)()  # Create a ctypes array of GLuints
  func(n, texture_ids)
  return list(texture_ids)


# glUniform1i(GLint location, GLint v0)
def glUniform1i(location: GLint, v0: GLint) -> None:
  """Specify the value of a uniform variable for the current program object."""
  func = get_gl_function("glUniform1i", None, [GLint, GLint])
  func(location, v0)


# glTexImage2D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const GLvoid *pixels)
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
  """Specify a two-dimensional texture image."""
  func = get_gl_function("glTexImage2D", None, [GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, GLvoidp])
  func(target, level, internalformat, width, height, border, _format, _type, pixels)


# glBindTexture(GLenum target, GLuint texture)
def glBindTexture(target: GLenum, texture: GLuint) -> None:
  """Bind a named texture to a texturing target."""
  func = get_gl_function("glBindTexture", None, [GLenum, GLuint])
  func(target, texture)


# glTexParameteri(GLenum target, GLenum pname, GLint param)
def glTexParameteri(target: GLenum, pname: GLenum, param: GLint) -> None:
  """Set texture parameters (integer)."""
  func = get_gl_function("glTexParameteri", None, [GLenum, GLenum, GLint])
  func(target, pname, param)


# glTexParameterf(GLenum target, GLenum pname, GLfloat param)
def glTexParameterf(target: GLenum, pname: GLenum, param: GLfloat) -> None:
  """Set texture parameters (float)."""
  func = get_gl_function("glTexParameterf", None, [GLenum, GLenum, GLfloat])
  func(target, pname, param)


# glActiveTexture(GLenum texture)
def glActiveTexture(texture: GLenum) -> None:
    """Select active texture unit."""
    func = get_gl_function("glActiveTexture", None, [GLenum])
    func(texture)


# --- OpenGL Constants ---

# Pixel storage parameters
GL_UNPACK_ALIGNMENT = cast(GLenum, 0x0CF5)
GL_UNPACK_ROW_LENGTH = cast(GLenum, 0x0CF2)

# Texture targets
GL_TEXTURE_2D = cast(GLenum, 0x0DE1)

# Texture units
GL_TEXTURE0 = cast(GLenum, 0x84C0)
# Add GL_TEXTURE1, GL_TEXTURE2 etc. if needed (GL_TEXTURE0 + i)

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
GL_RGBA = cast(GLenum, 0x1908)

# Pixel types (type for TexImage*)
GL_UNSIGNED_BYTE = cast(GLenum, 0x1401)
