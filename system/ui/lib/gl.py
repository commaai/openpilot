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


# void glPixelStorei(GLenum pname, GLint param)
def glPixelStorei(pname: GLenum, param: GLint) -> None:
  func = get_gl_function("glPixelStorei", None, [GLenum, GLint])
  func(pname, param)


# void glGenTextures(GLsizei n, GLuint *textures)
def glGenTextures(n: GLsizei) -> list[GLuint]:
  func = get_gl_function("glGenTextures", None, [GLsizei, ctypes.POINTER(GLuint)])
  texture_ids = (GLuint * n)()  # Create a ctypes array of GLuints
  func(n, texture_ids)
  return list(texture_ids)


# void glUniform1i(GLint location, GLint v0)
def glUniform1i(location: GLint, v0: GLint) -> None:
  func = get_gl_function("glUniform1i", None, [GLint, GLint])
  func(location, v0)


# void glTexImage2D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const GLvoid *pixels)
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


# void glTexParameteri(GLenum target, GLenum pname, GLint param)
def glTexParameteri(target: GLenum, pname: GLenum, param: GLint) -> None:
  func = get_gl_function("glTexParameteri", None, [GLenum, GLenum, GLint])
  func(target, pname, param)


# void glTexParameterf(GLenum target, GLenum pname, GLfloat param)
def glTexParameterf(target: GLenum, pname: GLenum, param: GLfloat) -> None:
  func = get_gl_function("glTexParameterf", None, [GLenum, GLenum, GLfloat])
  func(target, pname, param)


# void glActiveTexture(GLenum texture)
def glActiveTexture(texture: GLenum) -> None:
  func = get_gl_function("glActiveTexture", None, [GLenum])
  func(texture)


# void glDeleteVertexArrays(GLsizei n, const GLuint *arrays)
def glDeleteVertexArrays(n: GLsizei, arrays: list[GLuint]) -> None:
  func = get_gl_function("glDeleteVertexArrays", None, [GLsizei, ctypes.POINTER(GLuint)])
  arr = (GLuint * n)(*arrays)
  func(n, arr)


# void glDeleteBuffers(GLsizei n, const GLuint *buffers)
def glDeleteBuffers(n: GLsizei, buffers: list[GLuint]) -> None:
  func = get_gl_function("glDeleteBuffers", None, [GLsizei, ctypes.POINTER(GLuint)])
  arr = (GLuint * n)(*buffers)
  func(n, arr)


# void glDeleteTextures(GLsizei n, const GLuint *textures)
def glDeleteTextures(n: GLsizei, textures: list[GLuint]) -> None:
  func = get_gl_function("glDeleteTextures", None, [GLsizei, ctypes.POINTER(GLuint)])
  arr = (GLuint * n)(*textures)
  func(n, arr)


# void glGenVertexArrays(GLsizei n, GLuint *arrays)
def glGenVertexArrays(n: GLsizei) -> list[GLuint]:
  func = get_gl_function("glGenVertexArrays", None, [GLsizei, ctypes.POINTER(GLuint)])
  arrays = (GLuint * n)()
  func(n, arrays)
  return list(arrays)


# void glBindVertexArray(GLuint array)
def glBindVertexArray(array: GLuint) -> None:
  func = get_gl_function("glBindVertexArray", None, [GLuint])
  func(array)


# void glGenBuffers(GLsizei n, GLuint *buffers)
def glGenBuffers(n: GLsizei) -> list[GLuint]:
  func = get_gl_function("glGenBuffers", None, [GLsizei, ctypes.POINTER(GLuint)])
  buffers = (GLuint * n)()
  func(n, buffers)
  return list(buffers)


# void glBindBuffer(GLenum target, GLuint buffer)
def glBindBuffer(target: GLenum, buffer: GLuint) -> None:
  func = get_gl_function("glBindBuffer", None, [GLenum, GLuint])
  func(target, buffer)


# void glBufferData(GLenum target, GLsizeiptr size, const void *data, GLenum usage)
def glBufferData(target: GLenum, size: GLsizeiptr, data: ctypes.c_void_p | None, usage: GLenum) -> None:
  func = get_gl_function("glBufferData", None, [GLenum, GLsizeiptr, GLvoidp, GLenum])
  func(target, size, data, usage)


# void glEnableVertexAttribArray(GLuint index)
def glEnableVertexAttribArray(index: GLuint) -> None:
  func = get_gl_function("glEnableVertexAttribArray", None, [GLuint])
  func(index)


# void glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer)
def glVertexAttribPointer(
  index: GLuint,
  size: GLint,
  _type: GLenum,
  normalized: GLboolean,
  stride: GLsizei,
  pointer: ctypes.c_void_p,
) -> None:
  func = get_gl_function(
    "glVertexAttribPointer",
    None,
    [GLuint, GLint, GLenum, GLboolean, GLsizei, GLvoidp],
  )
  func(index, size, _type, normalized, stride, pointer)


# void glUseProgram(GLuint program)
def glUseProgram(program: GLuint) -> None:
  func = get_gl_function("glUseProgram", None, [GLuint])
  func(program)


# void glClearColor(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha)
def glClearColor(red: GLfloat, green: GLfloat, blue: GLfloat, alpha: GLfloat) -> None:
  func = get_gl_function("glClearColor", None, [GLfloat, GLfloat, GLfloat, GLfloat])
  func(red, green, blue, alpha)


# void glClear(GLbitfield mask)
def glClear(mask: GLbitfield) -> None:
  func = get_gl_function("glClear", None, [GLbitfield])
  func(mask)


# void glViewport(GLint x, GLint y, GLsizei width, GLsizei height)
def glViewport(x: GLint, y: GLint, width: GLsizei, height: GLsizei) -> None:
  func = get_gl_function("glViewport", None, [GLint, GLint, GLsizei, GLsizei])
  func(x, y, width, height)


# GLenum glGetError(void)
def glGetError() -> GLenum:
  func = get_gl_function("glGetError", GLenum, [])
  return func()


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


# void glDrawElements(GLenum mode, GLsizei count, GLenum type, const void *indices)
def glDrawElements(mode: GLenum, count: GLsizei, _type: GLenum, indices: ctypes.c_void_p | None) -> None:
  func = get_gl_function("glDrawElements", None, [GLenum, GLsizei, GLenum, GLvoidp])
  func(mode, count, _type, indices)


# void glDisableVertexAttribArray(GLuint index)
def glDisableVertexAttribArray(index: GLuint) -> None:
  func = get_gl_function("glDisableVertexAttribArray", None, [GLuint])
  func(index)


# GLuint glCreateShader(GLenum shaderType)
def glCreateShader(shaderType: GLenum) -> GLuint:
  func = get_gl_function("glCreateShader", GLuint, [GLenum])
  return func(shaderType)


# void glShaderSource(GLuint shader, GLsizei count, const GLchar *const* string, const GLint *length)
def glShaderSource(shader: GLuint, source: str) -> None:
  func = get_gl_function(
    "glShaderSource",
    None,
    [GLuint, GLsizei, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(GLint)],
  )
  # Prepare source as bytes
  source_bytes = source.encode("utf-8")
  src_ptr = ctypes.c_char_p(source_bytes)
  count = 1
  length = GLint(len(source_bytes))
  func(shader, count, ctypes.byref(src_ptr), ctypes.byref(length))


# void glCompileShader(GLuint shader)
def glCompileShader(shader: GLuint) -> None:
  func = get_gl_function("glCompileShader", None, [GLuint])
  func(shader)


# void glGetShaderiv(GLuint shader, GLenum pname, GLint *params)
def glGetShaderiv(shader: GLuint, pname: GLenum) -> GLint:
  func = get_gl_function("glGetShaderiv", None, [GLuint, GLenum, ctypes.POINTER(GLint)])
  params = GLint()
  func(shader, pname, ctypes.byref(params))
  return params.value


# void glGetShaderInfoLog(GLuint shader, GLsizei maxLength, GLsizei *length, GLchar *infoLog)
def glGetShaderInfoLog(shader: GLuint) -> str:
  func = get_gl_function(
    "glGetShaderInfoLog",
    None,
    [GLuint, GLsizei, ctypes.POINTER(GLsizei), ctypes.c_char_p],
  )
  max_length = 1024
  buffer = ctypes.create_string_buffer(max_length)
  length = GLsizei()
  func(shader, max_length, ctypes.byref(length), buffer)
  return buffer.value[: length.value].decode("utf-8")


# GLuint glCreateProgram(void)
def glCreateProgram() -> GLuint:
  func = get_gl_function("glCreateProgram", GLuint, [])
  return func()


# void glAttachShader(GLuint program, GLuint shader)
def glAttachShader(program: GLuint, shader: GLuint) -> None:
  func = get_gl_function("glAttachShader", None, [GLuint, GLuint])
  func(program, shader)


# void glLinkProgram(GLuint program)
def glLinkProgram(program: GLuint) -> None:
  func = get_gl_function("glLinkProgram", None, [GLuint])
  func(program)


# void glGetProgramiv(GLuint program, GLenum pname, GLint *params)
def glGetProgramiv(program: GLuint, pname: GLenum) -> GLint:
  func = get_gl_function("glGetProgramiv", None, [GLuint, GLenum, ctypes.POINTER(GLint)])
  params = GLint()
  func(program, pname, ctypes.byref(params))
  return params.value


# void glGetProgramInfoLog(GLuint program, GLsizei maxLength, GLsizei *length, GLchar *infoLog)
def glGetProgramInfoLog(program: GLuint) -> str:
  func = get_gl_function(
    "glGetProgramInfoLog",
    None,
    [GLuint, GLsizei, ctypes.POINTER(GLsizei), ctypes.c_char_p],
  )
  max_length = 1024
  buffer = ctypes.create_string_buffer(max_length)
  length = GLsizei()
  func(program, max_length, ctypes.byref(length), buffer)
  return buffer.value[: length.value].decode("utf-8")


# void glDeleteShader(GLuint shader)
def glDeleteShader(shader: GLuint) -> None:
  func = get_gl_function("glDeleteShader", None, [GLuint])
  func(shader)


# GLint glGetUniformLocation(GLuint program, const GLchar *name)
def glGetUniformLocation(program: GLuint, name: str) -> GLint:
  func = get_gl_function("glGetUniformLocation", GLint, [GLuint, ctypes.c_char_p])
  name_bytes = name.encode("utf-8")
  return func(program, name_bytes)


# GLint glGetAttribLocation(GLuint program, const GLchar *name)
def glGetAttribLocation(program: GLuint, name: str) -> GLint:
  func = get_gl_function("glGetAttribLocation", GLint, [GLuint, ctypes.c_char_p])
  name_bytes = name.encode("utf-8")
  return func(program, name_bytes)


# --- OpenGL Constants ---

GL_NO_ERROR = cast(GLenum, 0)

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

# Framebuffer
GL_STENCIL_BUFFER_BIT = cast(GLbitfield, 0x00000400)
GL_COLOR_BUFFER_BIT = cast(GLbitfield, 0x00004000)

# Drawing
GL_TRIANGLES = cast(GLenum, 0x0004)

# Buffer
GL_ARRAY_BUFFER = cast(GLenum, 0x8892)
GL_ELEMENT_ARRAY_BUFFER = cast(GLenum, 0x8893)
GL_STATIC_DRAW = cast(GLenum, 0x88E4)

# Shaders
GL_VERTEX_SHADER = cast(GLenum, 0x8B31)
GL_FRAGMENT_SHADER = cast(GLenum, 0x8B30)
GL_COMPILE_STATUS = cast(GLenum, 0x8B81)
GL_LINK_STATUS = cast(GLenum, 0x8B82)
