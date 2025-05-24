import os
import cffi
from typing import Any

HAS_EGL = False
_ffi = None
_egl = None
_gles = None

# EGL constants
EGL_NO_CONTEXT = None
EGL_NO_DISPLAY = None
EGL_NO_IMAGE_KHR = None
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

# Initialize EGL and load necessary functions
try:
  _ffi = cffi.FFI()
  _ffi.cdef("""
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
  _egl = _ffi.dlopen("libEGL.so")
  _gles = _ffi.dlopen("libGLESv2.so")

  # Cast NULL pointers
  EGL_NO_CONTEXT = _ffi.cast("void *", 0)
  EGL_NO_DISPLAY = _ffi.cast("void *", 0)
  EGL_NO_IMAGE_KHR = _ffi.cast("void *", 0)

  # Bind functions
  eglGetCurrentDisplay = _egl.eglGetCurrentDisplay
  eglCreateImageKHR = _egl.eglCreateImageKHR
  eglDestroyImageKHR = _egl.eglDestroyImageKHR
  glEGLImageTargetTexture2DOES = _gles.glEGLImageTargetTexture2DOES
  eglGetError = _egl.eglGetError
  glBindTexture = _gles.glBindTexture
  glActiveTexture = _gles.glActiveTexture

  HAS_EGL = True
except Exception as e:
  print(f"EGL support disabled: {e}")
  HAS_EGL = False


def create_egl_image(
  egl_display, width: int, height: int, stride: int, fd: int, uv_offset: int
) -> dict[str, Any] | None:
  if not HAS_EGL or egl_display == EGL_NO_DISPLAY or _ffi is None:
    return None

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

  attr_array = _ffi.new("int[]", img_attrs)
  egl_image = eglCreateImageKHR(egl_display, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, _ffi.NULL, attr_array)

  if egl_image == EGL_NO_IMAGE_KHR:
    print(f"Failed to create EGL image: {eglGetError()}")
    os.close(dup_fd)
    return None

  return {"egl_image": egl_image, "fd": dup_fd}


def destroy_egl_image(egl_display, data: dict[str, Any]) -> None:
  if not HAS_EGL or egl_display == EGL_NO_DISPLAY or not data:
    return

  if "egl_image" in data and data["egl_image"]:
    eglDestroyImageKHR(egl_display, data["egl_image"])

  # Close the duplicated fd we created in create_egl_image()
  # We need to handle OSError since the fd might already be closed
  if "fd" in data and data["fd"]:
    try:
      os.close(data["fd"])
    except OSError:
      pass

def bind_egl_image_to_texture(texture_id: int, egl_image) -> None:
  if not HAS_EGL or not egl_image:
    return

  glActiveTexture(GL_TEXTURE0)
  glBindTexture(GL_TEXTURE_EXTERNAL_OES, texture_id)
  glEGLImageTargetTexture2DOES(GL_TEXTURE_EXTERNAL_OES, egl_image)
