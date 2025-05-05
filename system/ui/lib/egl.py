import ctypes
import sys
from collections.abc import Callable
from typing import Any, cast

# --- Type Definitions ---
EGLBoolean = ctypes.c_uint
EGLenum = ctypes.c_uint
EGLint = ctypes.c_int
EGLDisplay = ctypes.c_void_p
EGLContext = ctypes.c_void_p
EGLImageKHR = ctypes.c_void_p
EGLClientBuffer = ctypes.c_void_p

# for glEGLImageTargetTexture2DOES
GLenum = ctypes.c_uint

# --- dlsym Setup ---
if sys.platform.startswith("linux") or sys.platform == "darwin":
  try:
    ctypes.pythonapi.dlsym.restype = ctypes.c_void_p
    ctypes.pythonapi.dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    RTLD_DEFAULT = ctypes.c_void_p(0)
  except AttributeError as err:
    raise ImportError("Cannot access ctypes.pythonapi.dlsym. Ensure you are on a POSIX-like system.") from err
else:
  raise ImportError(f"Platform '{sys.platform}' not supported by egl.py.")

# --- Generic Function Loader ---
_egl_func_cache: dict[str, Callable[..., Any]] = {}


def get_egl_function(
  name: str,
  restype: Any | None,
  argtypes: list[Any],
) -> Callable[..., Any]:
  """
  Load an EGL (or related GL) function by name via dlsym, cache it,
  and return a callable with the requested signature.
  """
  if name in _egl_func_cache:
    return _egl_func_cache[name]
  addr = ctypes.pythonapi.dlsym(RTLD_DEFAULT, name.encode("utf-8"))
  if not addr:
    raise AttributeError(f"Function '{name}' not found via dlsym. Ensure an EGL/GL context is active and the extension is supported.")
  proto = ctypes.CFUNCTYPE(restype, *argtypes)
  func = cast(Callable[..., Any], proto(addr))
  _egl_func_cache[name] = func
  return func


# --- EGL Function Wrappers ---


def eglGetCurrentDisplay() -> EGLDisplay:
  func = get_egl_function("eglGetCurrentDisplay", EGLDisplay, [])
  return func()


def eglGetError() -> EGLint:
  func = get_egl_function("eglGetError", EGLint, [])
  return func()


def eglDestroyImageKHR(dpy: EGLDisplay, image: EGLImageKHR) -> EGLBoolean:
  func = get_egl_function(
    "eglDestroyImageKHR",
    EGLBoolean,
    [EGLDisplay, EGLImageKHR],
  )
  return func(dpy, image)


def eglCreateImageKHR(
  dpy: EGLDisplay,
  ctx: EGLContext,
  target: EGLenum,
  buffer: EGLClientBuffer,
  attrib_list: list[EGLint],
) -> EGLImageKHR:
  func = get_egl_function(
    "eglCreateImageKHR",
    EGLImageKHR,
    [
      EGLDisplay,
      EGLContext,
      EGLenum,
      EGLClientBuffer,
      ctypes.POINTER(EGLint),
    ],
  )
  # NULL-terminated attribute list
  attribs = (EGLint * (len(attrib_list) + 1))(*attrib_list, EGL_NONE)
  return func(dpy, ctx, target, buffer, attribs)


def glEGLImageTargetTexture2DOES(target: GLenum, image: EGLImageKHR) -> None:
  func = get_egl_function(
    "glEGLImageTargetTexture2DOES",
    None,
    [GLenum, ctypes.c_void_p],
  )
  func(target, image)


# --- EGL Constants ---
EGL_NO_DISPLAY = cast(EGLDisplay, 0)
EGL_NO_CONTEXT = cast(EGLContext, 0)
EGL_NONE = cast(EGLint, 0x3038)
EGL_SUCCESS = cast(EGLint, 0x3000)

EGL_LINUX_DMA_BUF_EXT = cast(EGLenum, 0x3270)
EGL_LINUX_DRM_FOURCC_EXT = cast(EGLenum, 0x3271)
EGL_DMA_BUF_PLANE0_FD_EXT = cast(EGLenum, 0x3272)
EGL_DMA_BUF_PLANE0_OFFSET_EXT = cast(EGLenum, 0x3273)
EGL_DMA_BUF_PLANE0_PITCH_EXT = cast(EGLenum, 0x3274)
EGL_DMA_BUF_PLANE1_FD_EXT = cast(EGLenum, 0x3275)
EGL_DMA_BUF_PLANE1_OFFSET_EXT = cast(EGLenum, 0x3276)
EGL_DMA_BUF_PLANE1_PITCH_EXT = cast(EGLenum, 0x3277)

EGL_WIDTH = cast(EGLint, 0x3057)
EGL_HEIGHT = cast(EGLint, 0x3056)

# DRM FourCC for NV12
DRM_FORMAT_NV12 = cast(EGLint, 0x3231564E)

# For binding the image as an external texture
GL_TEXTURE_EXTERNAL_OES = cast(GLenum, 0x8D65)
