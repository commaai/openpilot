import ctypes
import os
import threading
import time
from collections import deque

import numpy as np
import raylib as rl
from raylib import ffi
from raylib.defines import RL_FLOAT
from raylib.enums import PixelFormat

from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware import APPLE, TICI
from openpilot.system.ui.lib import egl, gl
from openpilot.system.ui.lib.application import gui_app

VERTEX_SHADER = f"""{'#version 330 core' if APPLE else '#version 300 es'}
layout(location = 0) in vec4 aPosition;
layout(location = 1) in vec2 aTexCoord;
uniform mat4 uTransform;
out vec2 vTexCoord;
void main() {{
  gl_Position = uTransform * aPosition;
  vTexCoord = aTexCoord;
}}
"""

QCOM2_FRAGMENT_SHADER = """#version 300 es
#extension GL_OES_EGL_image_external_essl3 : enable
precision mediump float;
uniform samplerExternalOES uTexture;
in vec2 vTexCoord;
out vec4 colorOut;
void main() {
  colorOut = texture(uTexture, vTexCoord);
  // gamma to improve worst case visibility when dark
  colorOut.rgb = pow(colorOut.rgb, vec3(1.0/1.28));
}"""

PC_FRAGMENT_SHADER = f"""#version {'330 core' if APPLE else '300 es'}
{'precision mediump float;' if not APPLE else ''}
uniform sampler2D uTextureY;
uniform sampler2D uTextureUV;
in vec2 vTexCoord;
out vec4 colorOut;
void main() {{
  float y = texture(uTextureY, vTexCoord).r;
  vec2 uv = texture(uTextureUV, vTexCoord).rg - 0.5;
  float r = y + 1.402 * uv.y;
  float g = y - 0.344 * uv.x - 0.714 * uv.y;
  float b = y + 1.772 * uv.x;
  colorOut = vec4(r, g, b, 1.0);
}}
"""

FRAGMENT_SHADER = QCOM2_FRAGMENT_SHADER if TICI else PC_FRAGMENT_SHADER

FRAME_BUFFER_SIZE = 5


class CameraView:
  def __init__(self, stream_name: str, stream_type: VisionStreamType):
    self.stream_name = stream_name
    self.active_stream_type = stream_type
    self.requested_stream_type = stream_type

    self.frames: deque[tuple[int, VisionBuf]] = deque()
    self.frame_lock = threading.Lock()
    self.prev_frame_id = 0

    self.stream_width = 0
    self.stream_height = 0
    self.stream_stride = 0

    self._shader: rl.Shader | None = None
    self._vao: int = 0
    self._vbo: int = 0
    self._ebo: int = 0
    self._textures: list[int] | None = None  # Y and UV textures
    self._egl_images: dict[int, egl.EGLImageKHR] = {}

    self.vipc_client: VisionIpcClient | None = None
    self.vipc_thread = threading.Thread(target=self._vipc_thread_func, daemon=True)
    self.vipc_thread_stop_event = threading.Event()
    self.vipc_connected_event = threading.Event()

    self._initialize_gl()
    self.vipc_thread.start()

  def _initialize_gl(self) -> None:
    self._shader = rl.LoadShaderFromMemory(VERTEX_SHADER.encode("utf-8"), FRAGMENT_SHADER.encode("utf-8"))
    position_loc = rl.rlGetLocationAttrib(self._shader.id, b"aPosition")
    texcoord_loc = rl.rlGetLocationAttrib(self._shader.id, b"aTexCoord")

    if self.requested_stream_type == VisionStreamType.VISION_STREAM_DRIVER:
      x1, x2, y1, y2 = 0.0, 1.0, 1.0, 0.0
    else:
      x1, x2, y1, y2 = 1.0, 0.0, 1.0, 0.0

    frame_coords = np.array([
      -1.0, -1.0, x2, y1,
      -1.0,  1.0, x2, y2,
       1.0,  1.0, x1, y2,
       1.0, -1.0, x1, y1,
    ], dtype=np.float32)  # fmt: skip
    frame_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)

    self._vao = rl.rlLoadVertexArray()
    rl.rlEnableVertexArray(self._vao)
    self._vbo = rl.rlLoadVertexBuffer(ffi.cast("void *", frame_coords.ctypes.data), frame_coords.nbytes, False)
    rl.rlEnableVertexAttribute(position_loc)
    rl.rlSetVertexAttribute(position_loc, 2, RL_FLOAT, False, 4 * 4, 0)
    rl.rlEnableVertexAttribute(texcoord_loc)
    rl.rlSetVertexAttribute(texcoord_loc, 2, RL_FLOAT, False, 4 * 4, 2 * 4)
    self._ebo = rl.rlLoadVertexBufferElement(ffi.cast("void *", frame_indices.ctypes.data), frame_indices.nbytes, False)

    rl.rlEnableShader(self._shader.id)
    if TICI:
      gl.glUniform1i(rl.rlGetLocationUniform(self._shader.id, b"uTexture"), 0)
    else:
      gl.glUniform1i(rl.rlGetLocationUniform(self._shader.id, b"uTextureY"), 0)
      gl.glUniform1i(rl.rlGetLocationUniform(self._shader.id, b"uTextureUV"), 1)

  def _vipc_thread_func(self) -> None:
    cur_stream = self.requested_stream_type
    self.vipc_client = VisionIpcClient(self.stream_name, cur_stream, False)

    while not self.vipc_thread_stop_event.is_set():
      if not self.vipc_client.is_connected() or cur_stream != self.requested_stream_type:
        cur_stream = self.requested_stream_type
        self.vipc_client = VisionIpcClient(self.stream_name, cur_stream, False)

      if not self.vipc_client.is_connected():
        streams = VisionIpcClient.available_streams(self.stream_name, False)
        if not streams:
          time.sleep(0.1)
          continue
        if not self.vipc_client.connect(True):
          time.sleep(0.1)
          continue

        self.vipc_connected_event.set()

      buf = self.vipc_client.recv(1000)
      if buf is not None:
        with self.frame_lock:
          self.frames.append((self.vipc_client.frame_id, buf))
          while len(self.frames) > FRAME_BUFFER_SIZE:
            self.frames.popleft()

  def _on_vipc_connected(self) -> None:
    if self.vipc_client is None:
      return

    self.stream_width = self.vipc_client.width or 0
    self.stream_height = self.vipc_client.height or 0
    self.stream_stride = self.vipc_client.stride or 0

    # Scale the frame to fit the widget while maintaining the aspect ratio.
    widget_aspect_ratio = gui_app.width / gui_app.height
    frame_aspect_ratio = self.stream_width / self.stream_height
    x = min(frame_aspect_ratio / widget_aspect_ratio, 1.0)
    y = min(widget_aspect_ratio / frame_aspect_ratio, 1.0)
    rl.rlSetUniformMatrix(rl.rlGetLocationUniform(self._shader.id, b"uTransform"), rl.MatrixScale(x, y, 1.0))

    if TICI:
      egl_display = egl.eglGetCurrentDisplay()
      assert egl_display != egl.EGL_NO_DISPLAY
      if self._egl_images is not None:
        for _, image in self._egl_images.items():
          egl.eglDestroyImageKHR(egl_display, image)
          assert egl.eglGetError() == egl.EGL_SUCCESS, egl.eglGetError()
        self._egl_images.clear()

      # import buffers into OpenGL
      for i in range(self.vipc_client.num_buffers):
        fd = os.dup(self.vipc_client.get_fd(i))  # eglDestroyImageKHR will close, so duplicate
        attrs = [
          egl.EGL_WIDTH, self.vipc_client.width,
          egl.EGL_HEIGHT, self.vipc_client.height,
          egl.EGL_LINUX_DRM_FOURCC_EXT, egl.DRM_FORMAT_NV12,
          egl.EGL_DMA_BUF_PLANE0_FD_EXT, fd,
          egl.EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
          egl.EGL_DMA_BUF_PLANE0_PITCH_EXT, self.vipc_client.stride,
          egl.EGL_DMA_BUF_PLANE1_FD_EXT, fd,
          egl.EGL_DMA_BUF_PLANE1_OFFSET_EXT, self.vipc_client.uv_offset,
          egl.EGL_DMA_BUF_PLANE1_PITCH_EXT, self.vipc_client.stride,
          egl.EGL_NONE,
        ]
        attrs_array = (ctypes.c_int * len(attrs))(*attrs)
        self._egl_images[i] = egl.eglCreateImageKHR(egl_display, egl.EGL_NO_CONTEXT, egl.EGL_LINUX_DMA_BUF_EXT, 0, attrs_array)
        assert egl.eglGetError() == egl.EGL_SUCCESS, egl.eglGetError()
    else:
      if self._textures is not None:
        for tex_id in self._textures:
          rl.rlUnloadTexture(tex_id)

      self._textures = [
        rl.rlLoadTexture(ffi.NULL, self.stream_width, self.stream_height, PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAYSCALE, 1),
        rl.rlLoadTexture(ffi.NULL, self.stream_width // 2, self.stream_height // 2, PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA, 0),
      ]

      # FIXME: why is this required
      gl.glBindTexture(gl.GL_TEXTURE_2D, self._textures[1])
      gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RG8, self.stream_width // 2, self.stream_height // 2, 0, gl.GL_RG, gl.GL_UNSIGNED_BYTE, None)
      gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

  def render(self) -> None:
    if self.vipc_connected_event.is_set():
      self.vipc_connected_event.clear()
      self._on_vipc_connected()

    if self._shader is None or self._vao <= 0 or (not TICI and self._textures is None):
      return

    with self.frame_lock:
      if not self.frames or len(self.frames) == 0:
        return

      frame_idx = len(self.frames) - 1
      frame_id, frame = self.frames[frame_idx]
      assert frame is not None

      # Log duplicate/dropped frames
      if frame_id == self.prev_frame_id:
        cloudlog.debug(f"Drawing same frame twice {frame_id}")
      elif frame_id != self.prev_frame_id + 1:
        cloudlog.debug(f"Skipped frame {frame_id}")
      self.prev_frame_id = frame_id

      rl.rlEnableShader(self._shader.id)
      assert gl.glGetError() == gl.GL_NO_ERROR, gl.glGetError()
      rl.rlEnableVertexArray(self._vao)
      assert gl.glGetError() == gl.GL_NO_ERROR, gl.glGetError()
      gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
      assert gl.glGetError() == gl.GL_NO_ERROR, gl.glGetError()

      if TICI:
        # no frame copy
        rl.rlActiveTextureSlot(0)
        egl.glEGLImageTargetTexture2DOES(egl.GL_TEXTURE_EXTERNAL_OES, self._egl_images[frame.idx])
        gl_errno = gl.glGetError()
        egl_errno = egl.eglGetError()
        print("gl:", gl_errno, "egl:", egl_errno)
        assert egl_errno == egl.EGL_SUCCESS, egl_errno
        assert gl_errno == gl.GL_NO_ERROR, gl_errno
      else:
        # fallback to copy
        gl.glPixelStorei(gl.GL_UNPACK_ROW_LENGTH, self.stream_stride)
        rl.rlActiveTextureSlot(0)
        rl.rlUpdateTexture(self._textures[0], 0, 0, self.stream_width, self.stream_height,
                           PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAYSCALE, ffi.cast("void *", frame.y.ctypes.data))
        assert gl.glGetError() == gl.GL_NO_ERROR, gl.glGetError()

        gl.glPixelStorei(gl.GL_UNPACK_ROW_LENGTH, self.stream_stride // 2)
        rl.rlActiveTextureSlot(1)
        rl.rlUpdateTexture(self._textures[1], 0, 0, self.stream_width // 2, self.stream_height // 2,
                           PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA, ffi.cast("void *", frame.uv.ctypes.data))
        assert gl.glGetError() == gl.GL_NO_ERROR, gl.glGetError()

      rl.rlDrawVertexArrayElements(0, 6, ffi.NULL)

      rl.rlDisableVertexArray()

      gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
      rl.rlActiveTextureSlot(0)
      gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
      assert gl.glGetError() == gl.GL_NO_ERROR, gl.glGetError()
      gl.glPixelStorei(gl.GL_UNPACK_ROW_LENGTH, 0)
      assert gl.glGetError() == gl.GL_NO_ERROR, gl.glGetError()

      rl.rlDisableShader()

  def close(self) -> None:
    self.vipc_thread_stop_event.set()
    if self.vipc_thread and self.vipc_thread.is_alive():
      self.vipc_thread.join(timeout=2.0)
      if self.vipc_thread.is_alive():
        print("VIPC thread did not exit cleanly")

    if self._shader:
      rl.UnloadShader(self._shader)
      self._shader = None

    if self._vao > 0:
      rl.rlUnloadVertexArray(self._vao)
      self._vao = 0

    if self._vbo > 0:
      rl.rlUnloadVertexBuffer(self._vbo)
      self._vbo = 0

    if self._ebo > 0:
      rl.rlUnloadVertexBuffer(self._ebo)
      self._ebo = 0

    if self._textures:
      for tex_id in self._textures:
        rl.rlUnloadTexture(tex_id)
      self._textures = None

    if self._egl_images is not None:
      egl_display = egl.eglGetCurrentDisplay()
      assert egl_display != egl.EGL_NO_DISPLAY
      for _, image in self._egl_images.items():
        egl.eglDestroyImageKHR(egl_display, image)
        assert egl.eglGetError() == egl.EGL_SUCCESS, egl.eglGetError()
      self._egl_images.clear()


def run():
  gui_app.init_window("watch3")
  road_camera_view = CameraView("camerad", VisionStreamType.VISION_STREAM_ROAD)
  for _ in gui_app.render():
    road_camera_view.render()
  road_camera_view.close()
  gui_app.close()


if __name__ == "__main__":
  thread = threading.Thread(target=run, daemon=True)
  thread.start()
  thread.join()
