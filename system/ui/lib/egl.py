import os
import cffi
from dataclasses import dataclass
from typing import Any
from openpilot.common.swaglog import cloudlog

# EGL constants
EGL_LINUX_DMA_BUF_EXT = 0x3270
EGL_WIDTH = 0x3057
EGL_HEIGHT = 0x3056
EGL_LINUX_DRM_FOURCC_EXT = 0x3271
EGL_DMA_BUF_PLANE0_FD_EXT = 0x3272
EGL_DMA_BUF_PLANE0_OFFSET_EXT = 0x3273
EGL_DMA_BUF_PLANE0_PITCH_EXT = 0x3274
EGL_DMA_BUF_PLANE1_FD_EXT = 0x3275
EGL_DMA_BUF_PLANE1_OFFSET_EXT = 0x3276
EGL_DMA_BUF_PLANE1_PITCH_EXT = 0x3277
EGL_NONE = 0x3038
GL_TEXTURE0 = 0x84C0
GL_TEXTURE_EXTERNAL_OES = 0x8D65

# DRM Format for NV12
DRM_FORMAT_NV12 = 842094158


@dataclass
class EGLImage:
  """Container for EGL image and associated resources"""

  egl_image: Any
  fd: int


@dataclass
class EGLState:
  """Container for all EGL-related state"""

  initialized: bool = False
  ffi: Any = None
  egl_lib: Any = None
  gles_lib: Any = None

  # EGL display connection - shared across all users
  display: Any = None

  # Constants
  NO_CONTEXT: Any = None
  NO_DISPLAY: Any = None
  NO_IMAGE_KHR: Any = None

  # Function pointers
  get_current_display: Any = None
  create_image_khr: Any = None
  destroy_image_khr: Any = None
  image_target_texture: Any = None
  get_error: Any = None
  bind_texture: Any = None
  active_texture: Any = None


# Create a single instance of the state
_egl = EGLState()


def init_egl() -> bool:
  """Initialize EGL and load necessary functions"""
  global _egl

  # Don't re-initialize if already done
  if _egl.initialized:
    return True

  try:
    _egl.ffi = cffi.FFI()
    _egl.ffi.cdef("""
      typedef int EGLint;
      typedef unsigned int EGLBoolean;
      typedef unsigned int EGLenum;
      typedef unsigned int GLenum;
      typedef void *EGLContext;
      typedef void *EGLDisplay;
      typedef void *EGLClientBuffer;
      typedef void *EGLImageKHR;
      typedef void *GLeglImageOES;

      EGLDisplay eglGetCurrentDisplay(void);
      EGLint eglGetError(void);
      EGLImageKHR eglCreateImageKHR(EGLDisplay dpy, EGLContext ctx,
                                  EGLenum target, EGLClientBuffer buffer,
                                  const EGLint *attrib_list);
      EGLBoolean eglDestroyImageKHR(EGLDisplay dpy, EGLImageKHR image);
      void glEGLImageTargetTexture2DOES(GLenum target, GLeglImageOES image);
      void glBindTexture(GLenum target, unsigned int texture);
      void glActiveTexture(GLenum texture);
    """)

    # Load libraries
    _egl.egl_lib = _egl.ffi.dlopen("libEGL.so")
    _egl.gles_lib = _egl.ffi.dlopen("libGLESv2.so")

    # Cast NULL pointers
    _egl.NO_CONTEXT = _egl.ffi.cast("void *", 0)
    _egl.NO_DISPLAY = _egl.ffi.cast("void *", 0)
    _egl.NO_IMAGE_KHR = _egl.ffi.cast("void *", 0)

    # Bind functions
    _egl.get_current_display = _egl.egl_lib.eglGetCurrentDisplay
    _egl.create_image_khr = _egl.egl_lib.eglCreateImageKHR
    _egl.destroy_image_khr = _egl.egl_lib.eglDestroyImageKHR
    _egl.image_target_texture = _egl.gles_lib.glEGLImageTargetTexture2DOES
    _egl.get_error = _egl.egl_lib.eglGetError
    _egl.bind_texture = _egl.gles_lib.glBindTexture
    _egl.active_texture = _egl.gles_lib.glActiveTexture

    # Initialize EGL display once here
    _egl.display = _egl.get_current_display()
    if _egl.display == _egl.NO_DISPLAY:
      raise RuntimeError("Failed to get EGL display")

    _egl.initialized = True
    return True
  except Exception as e:
    cloudlog.exception(f"EGL initialization failed: {e}")
    _egl.initialized = False
    return False


def create_egl_image(width: int, height: int, stride: int, fd: int, uv_offset: int) -> EGLImage | None:
  assert _egl.initialized, "EGL not initialized"

  # Duplicate fd since EGL needs it
  dup_fd = os.dup(fd)

  # Create image attributes for EGL
  img_attrs = [
    EGL_WIDTH, width,
    EGL_HEIGHT, height,
    EGL_LINUX_DRM_FOURCC_EXT, DRM_FORMAT_NV12,
    EGL_DMA_BUF_PLANE0_FD_EXT, dup_fd,
    EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
    EGL_DMA_BUF_PLANE0_PITCH_EXT, stride,
    EGL_DMA_BUF_PLANE1_FD_EXT, dup_fd,
    EGL_DMA_BUF_PLANE1_OFFSET_EXT, uv_offset,
    EGL_DMA_BUF_PLANE1_PITCH_EXT, stride,
    EGL_NONE
  ]

  attr_array = _egl.ffi.new("int[]", img_attrs)
  egl_image = _egl.create_image_khr(_egl.display, _egl.NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, _egl.ffi.NULL, attr_array)

  if egl_image == _egl.NO_IMAGE_KHR:
    cloudlog.error(f"Failed to create EGL image: {_egl.get_error()}")
    os.close(dup_fd)
    return None

  return EGLImage(egl_image=egl_image, fd=dup_fd)


def destroy_egl_image(egl_image: EGLImage) -> None:
  assert _egl.initialized, "EGL not initialized"

  _egl.destroy_image_khr(_egl.display, egl_image.egl_image)

  # Close the duplicated fd we created in create_egl_image()
  # We need to handle OSError since the fd might already be closed
  try:
    os.close(egl_image.fd)
  except OSError:
    pass


def bind_egl_image_to_texture(texture_id: int, egl_image: EGLImage) -> None:
  assert _egl.initialized, "EGL not initialized"

  _egl.active_texture(GL_TEXTURE0)
  _egl.bind_texture(GL_TEXTURE_EXTERNAL_OES, texture_id)
  _egl.image_target_texture(GL_TEXTURE_EXTERNAL_OES, egl_image.egl_image)
