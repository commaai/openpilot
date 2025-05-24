import os
import pyray as rl
from typing import Any
from openpilot.system.hardware import TICI
from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.system.ui.lib.application import gui_app

# Define EGL constants and functions using CFFI
try:
  import cffi

  _ffi = cffi.FFI()
  _ffi.cdef("""
    typedef int EGLint;
    typedef unsigned int EGLBoolean;
    typedef unsigned int EGLenum;
    typedef unsigned int GLenum;
    typedef void *EGLConfig;
    typedef void *EGLContext;
    typedef void *EGLDisplay;
    typedef void *EGLSurface;
    typedef void *EGLClientBuffer;
    typedef void *EGLImage;
    typedef void *EGLImageKHR;
    typedef void *GLeglImageOES;  // Added missing type definition

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
  try:
    _egl = _ffi.dlopen("libEGL.so")
    _gles = _ffi.dlopen("libGLESv2.so")

    # Define constants (normally in header files)
    EGL_NO_CONTEXT = _ffi.cast("void *", 0)
    EGL_NO_DISPLAY = _ffi.cast("void *", 0)
    EGL_NO_IMAGE_KHR = _ffi.cast("void *", 0)
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
    TEXTURE_EXTERNAL_OES = 0x8D65

    # OpenGL constants
    GL_TEXTURE0 = 0x84C0
    GL_TEXTURE_EXTERNAL_OES = TEXTURE_EXTERNAL_OES

    # Set up function bindings
    eglGetCurrentDisplay = _egl.eglGetCurrentDisplay
    eglCreateImageKHR = _egl.eglCreateImageKHR
    eglDestroyImageKHR = _egl.eglDestroyImageKHR
    glEGLImageTargetTexture2DOES = _gles.glEGLImageTargetTexture2DOES
    eglGetError = _egl.eglGetError
    glBindTexture = _gles.glBindTexture
    glActiveTexture = _gles.glActiveTexture

    HAS_EGL = True
  except (OSError, AttributeError) as e:
    print(f"Failed to load EGL libraries: {e}")
    HAS_EGL = False
except ImportError:
  print("CFFI not available, EGL support disabled")
  HAS_EGL = False

# DRM Format for NV12
DRM_FORMAT_NV12 = 842094158  # value for the NV12 format in DRM


VERTEX_SHADER = """
#version 300 es
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;
uniform mat4 mvp;
out vec2 fragTexCoord;
out vec4 fragColor;
void main() {
  fragTexCoord = vertexTexCoord;
  fragColor = vertexColor;
  gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""


if TICI and HAS_EGL:
  FRAME_FRAGMENT_SHADER = """
    #version 300 es
    #extension GL_OES_EGL_image_external_essl3 : enable
    precision mediump float;
    in vec2 fragTexCoord;
    uniform samplerExternalOES texture0;
    out vec4 fragColor;
    void main() {
      vec4 color = texture(texture0, fragTexCoord);
      fragColor = vec4(pow(color.rgb, vec3(1.0/1.28)), color.a);
    }
    """
else:
  FRAME_FRAGMENT_SHADER = """
    #version 300 es
    precision mediump float;
    in vec2 fragTexCoord;
    uniform sampler2D texture0;
    uniform sampler2D texture1;
    out vec4 fragColor;
    void main() {
      float y = texture(texture0, fragTexCoord).r;
      vec2 uv = texture(texture1, fragTexCoord).ra - 0.5;
      fragColor = vec4(y + 1.402*uv.y, y - 0.344*uv.x - 0.714*uv.y, y + 1.772*uv.x, 1.0);
    }
    """


class CameraView:
  def __init__(self, name: str, stream_type: VisionStreamType):
    self.client = VisionIpcClient(name, stream_type, False)
    self.shader = rl.load_shader_from_memory(VERTEX_SHADER, FRAME_FRAGMENT_SHADER)

    self.texture_y: rl.Texture | None = None
    self.texture_uv: rl.Texture | None = None
    self.egl_textures: dict[int, dict[str, Any]] = {}  # For EGL implementation
    self.frame: VisionBuf | None = None
    self.use_egl = TICI and HAS_EGL
    self.egl_texture: rl.Texture | None = None  # Single texture for all EGL images

    # For EGL
    if self.use_egl:
      self.egl_display = None
      self.setup_egl()

      # Create a single texture for EGL images
      temp_image = rl.gen_image_color(1, 1, rl.BLACK)
      self.egl_texture = rl.load_texture_from_image(temp_image)
      rl.unload_image(temp_image)

  def setup_egl(self):
    if not self.use_egl:
      return

    # Get current EGL display
    self.egl_display = eglGetCurrentDisplay()
    if self.egl_display == EGL_NO_DISPLAY:
      print("Warning: No EGL display available, falling back to texture copy")
      self.use_egl = False

  def create_egl_image(self, buffer_idx, width, height, stride, fd, uv_offset):
    """Create EGL image from DMA buffer - more efficient version"""
    if not self.use_egl or not self.egl_display:
      return None

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
    egl_image = eglCreateImageKHR(self.egl_display, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, _ffi.NULL, attr_array)

    if egl_image == EGL_NO_IMAGE_KHR:
      print(f"Failed to create EGL image: {eglGetError()}")
      os.close(dup_fd)
      return None

    return {"egl_image": egl_image, "fd": dup_fd}

  def close(self):
    self._clear_textures()

    # Clean up EGL resources
    if self.use_egl and self.egl_display:
      for data in self.egl_textures.values():
        if data["egl_image"]:
          eglDestroyImageKHR(self.egl_display, data["egl_image"])
        if data["fd"] > 0:
          os.close(data["fd"])

      # Clean up the single EGL texture
      if self.egl_texture and self.egl_texture.id:
        rl.unload_texture(self.egl_texture)
        self.egl_texture = None

    if self.shader and self.shader.id:
      rl.unload_shader(self.shader)

  def render(self, rect: rl.Rectangle):
    if not self._ensure_connection():
      return

    buffer = self.client.recv(timeout_ms=0)
    if buffer:
      self.frame = buffer

    if not self.frame:
      return

    # Calculate scaling to maintain aspect ratio
    scale = min(rect.width / self.frame.width, rect.height / self.frame.height)
    x_offset = rect.x + (rect.width - (self.frame.width * scale)) / 2
    y_offset = rect.y + (rect.height - (self.frame.height * scale)) / 2
    src_rect = rl.Rectangle(0, 0, float(self.frame.width), float(self.frame.height))
    dst_rect = rl.Rectangle(x_offset, y_offset, self.frame.width * scale, self.frame.height * scale)

    # Render with appropriate method
    if self.use_egl:
      self._render_egl(src_rect, dst_rect)
    else:
      self._render_textures(src_rect, dst_rect)

  def _render_egl(self, src_rect, dst_rect):
    """Render using EGL for direct buffer access - more efficient version"""
    idx = self.frame.idx

    # Create or get EGL image
    if idx not in self.egl_textures:
      self.egl_textures[idx] = self.create_egl_image(
        idx, self.frame.width, self.frame.height, self.frame.stride, self.frame.fd, self.frame.uv_offset
      )

    # If we have a valid EGL image, render it
    if idx in self.egl_textures and self.egl_textures[idx]:
      egl_image = self.egl_textures[idx]["egl_image"]

      # Update texture dimensions to match current frame
      self.egl_texture.width = self.frame.width
      self.egl_texture.height = self.frame.height

      # Activate texture and bind EGL image
      glActiveTexture(GL_TEXTURE0)
      glBindTexture(GL_TEXTURE_EXTERNAL_OES, self.egl_texture.id)
      glEGLImageTargetTexture2DOES(GL_TEXTURE_EXTERNAL_OES, egl_image)

      # Render with shader
      rl.begin_shader_mode(self.shader)
      rl.draw_texture_pro(self.egl_texture, src_rect, dst_rect, rl.Vector2(0, 0), 0.0, rl.WHITE)
      rl.end_shader_mode()

  def _render_textures(self, src_rect, dst_rect):
    """Render using texture copies (fallback method)"""
    if not self.texture_y or not self.texture_uv:
      return

    # Update textures with new frame data
    y_data = self.frame.data[: self.frame.uv_offset]
    uv_data = self.frame.data[self.frame.uv_offset :]

    rl.update_texture(self.texture_y, rl.ffi.cast("void *", y_data.ctypes.data))
    rl.update_texture(self.texture_uv, rl.ffi.cast("void *", uv_data.ctypes.data))

    # Render with shader
    rl.begin_shader_mode(self.shader)
    rl.set_shader_value_texture(self.shader, rl.get_shader_location(self.shader, "texture1"), self.texture_uv)
    rl.draw_texture_pro(self.texture_y, src_rect, dst_rect, rl.Vector2(0, 0), 0.0, rl.WHITE)
    rl.end_shader_mode()

  def _ensure_connection(self) -> bool:
    if not self.client.is_connected():
      self.frame = None
      if not self.client.connect(False) or not self.client.num_buffers:
        return False

      self._clear_textures()

      if not self.use_egl:
        self.texture_y = rl.load_texture_from_image(
          rl.Image(
            None, int(self.client.stride), int(self.client.height), 1, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAYSCALE
          )
        )
        self.texture_uv = rl.load_texture_from_image(
          rl.Image(
            None,
            int(self.client.stride // 2),
            int(self.client.height // 2),
            1,
            rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA,
          )
        )

    return True

  def _clear_textures(self):
    if self.texture_y and self.texture_y.id:
      rl.unload_texture(self.texture_y)
      self.texture_y = None

    if self.texture_uv and self.texture_uv.id:
      rl.unload_texture(self.texture_uv)
      self.texture_uv = None

    # Clean up EGL textures - more efficient version
    if self.use_egl and self.egl_display:
      for data in self.egl_textures.values():
        if data["egl_image"]:
          eglDestroyImageKHR(self.egl_display, data["egl_image"])
      self.egl_textures = {}


if __name__ == "__main__":
  gui_app.init_window("watch3")
  road_camera_view = CameraView("camerad", VisionStreamType.VISION_STREAM_ROAD)
  driver_camera_view = CameraView("camerad", VisionStreamType.VISION_STREAM_DRIVER)
  wide_road_camera_view = CameraView("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD)
  try:
    for _ in gui_app.render():
      road_camera_view.render(rl.Rectangle(gui_app.width // 4, 0, gui_app.width // 2, gui_app.height // 2))
      driver_camera_view.render(rl.Rectangle(0, gui_app.height // 2, gui_app.width // 2, gui_app.height // 2))
      wide_road_camera_view.render(
        rl.Rectangle(gui_app.width // 2, gui_app.height // 2, gui_app.width // 2, gui_app.height // 2)
      )
  finally:
    road_camera_view.close()
    driver_camera_view.close()
    wide_road_camera_view.close()