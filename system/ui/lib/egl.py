import ctypes.util
import os

# Load EGL library
try:
  egl_lib = ctypes.util.find_library('EGL')
  if egl_lib:
    egl = ctypes.CDLL(egl_lib)
  else:
    # Try common paths on Linux systems
    for path in ['/usr/lib/libEGL.so', '/usr/lib/aarch64-linux-gnu/libEGL.so']:
      if os.path.exists(path):
        egl = ctypes.CDLL(path)
        break
    else:
      raise ImportError("Could not find EGL library")
except ImportError as e:
  print(f"Error loading EGL library: {e}")

# Define necessary EGL constants
EGL_NO_CONTEXT = 0
EGL_NO_DISPLAY = 0
EGL_SUCCESS = 0x3000

# EGL extensions for DMA buffer handling
EGL_LINUX_DMA_BUF_EXT = 0x3270
EGL_LINUX_DRM_FOURCC_EXT = 0x3271
EGL_DMA_BUF_PLANE0_FD_EXT = 0x3272
EGL_DMA_BUF_PLANE0_OFFSET_EXT = 0x3273
EGL_DMA_BUF_PLANE0_PITCH_EXT = 0x3274
EGL_DMA_BUF_PLANE1_FD_EXT = 0x3275
EGL_DMA_BUF_PLANE1_OFFSET_EXT = 0x3276
EGL_DMA_BUF_PLANE1_PITCH_EXT = 0x3277
EGL_WIDTH = 0x3057
EGL_HEIGHT = 0x3056
EGL_NONE = 0x3038

# DRM format for NV12
DRM_FORMAT_NV12 = 842094158  # fourcc code for NV12

# OpenGL texture target for external textures
GL_TEXTURE_EXTERNAL_OES = 0x8D65

# Define function pointer types
EGLImageKHR = ctypes.c_void_p
EGLDisplay = ctypes.c_void_p
EGLContext = ctypes.c_void_p
EGLBoolean = ctypes.c_uint
EGLint = ctypes.c_int32

# Initialize extension functions
if egl is not None:
  try:
    # Get basic functions
    eglGetCurrentDisplay = egl.eglGetCurrentDisplay
    eglGetCurrentDisplay.restype = EGLDisplay

    eglGetError = egl.eglGetError
    eglGetError.restype = EGLint

    # Get extension functions via eglGetProcAddress
    _eglGetProcAddress = egl.eglGetProcAddress
    _eglGetProcAddress.argtypes = [ctypes.c_char_p]
    _eglGetProcAddress.restype = ctypes.c_void_p

    # Function pointers for extensions
    create_image_func = _eglGetProcAddress(b"eglCreateImageKHR")
    if create_image_func:
      PFNEGLCREATEIMAGEKHRPROC = ctypes.CFUNCTYPE(
        EGLImageKHR, EGLDisplay, EGLContext, ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(EGLint)
      )
      eglCreateImageKHR = PFNEGLCREATEIMAGEKHRPROC(create_image_func)

    destroy_image_func = _eglGetProcAddress(b"eglDestroyImageKHR")
    if destroy_image_func:
      PFNEGLDESTROYIMAGEKHRPROC = ctypes.CFUNCTYPE(EGLBoolean, EGLDisplay, EGLImageKHR)
      eglDestroyImageKHR = PFNEGLDESTROYIMAGEKHRPROC(destroy_image_func)

    target_texture_func = _eglGetProcAddress(b"glEGLImageTargetTexture2DOES")
    if target_texture_func:
      PFNGLEGLIMAGETARGETTEXTURE2DOESPROC = ctypes.CFUNCTYPE(None, ctypes.c_uint, EGLImageKHR)
      glEGLImageTargetTexture2DOES = PFNGLEGLIMAGETARGETTEXTURE2DOESPROC(target_texture_func)

  except Exception as e:
    print(f"Error initializing EGL functions: {e}")
