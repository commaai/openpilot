import threading
import time
from collections import deque
from ctypes import c_int

import numpy as np
import pyray as rl
from pyray import ffi, PixelFormat

from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware import APPLE
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib import gl

RL_UNSIGNED_BYTE = 0x1401  # GL_UNSIGNED_BYTE
RL_FLOAT = 0x1406  # GL_FLOAT


VERTEX_SHADER = f"""{'#version 330 core' if APPLE else '#version 300 es'}
layout(location = 0) in vec4 aPosition;
layout(location = 1) in vec2 aTexCoord;
out vec2 vTexCoord;
void main() {{
  gl_Position = aPosition;
  vTexCoord = aTexCoord;
}}
"""

FRAGMENT_SHADER = f"""#version {'330 core' if APPLE else '300 es'}
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

FRAME_BUFFER_SIZE = 5


class CheckGL:
  def __init__(self, name: str):
    self.name = name

  def __enter__(self):
    rl.rl_check_errors()
    print("\n>> ", self.name)

  def __exit__(self, exc_type, exc_val, exc_tb):
    rl.rl_check_errors()
    print("<< ", self.name)


class CameraView:
  def __init__(self, stream_name: str, stream_type: VisionStreamType):
    self.stream_name = stream_name
    self.active_stream_type = stream_type
    self.requested_stream_type = stream_type

    self.frames: deque[tuple[int, VisionBuf]] = deque()
    self.frame_lock = threading.Lock()
    self.prev_frame_id: int | None = None

    self.stream_width = 0
    self.stream_height = 0
    self.stream_stride = 0

    self.program: rl.Shader | None = None
    self.vao: int | None = None
    self.vbo: int | None = None
    self.ibo: int | None = None
    self.textures: tuple[int, int] | None = None  # Y and UV textures

    self.vipc_client: VisionIpcClient | None = None
    self.vipc_thread = threading.Thread(target=self._vipc_thread_func, daemon=True)
    self.vipc_thread_stop_event = threading.Event()
    self.vipc_connected_event = threading.Event()

    self._initialize_gl()
    self.vipc_thread.start()

  def _initialize_gl(self) -> None:
    self.program = rl.load_shader_from_memory(VERTEX_SHADER, FRAGMENT_SHADER)
    if not self.program:
      raise RuntimeError("Failed to load shader program")

    if self.requested_stream_type == VisionStreamType.VISION_STREAM_DRIVER:
      x1, x2, y1, y2 = 0.0, 1.0, 1.0, 0.0
    else:
      x1, x2, y1, y2 = 1.0, 0.0, 1.0, 0.0

    self.vao = rl.rl_load_vertex_array()
    rl.rl_enable_vertex_array(self.vao)

    vertices = np.array([
      -1.0, -1.0, x2, y1,
      -1.0,  1.0, x2, y2,
       1.0,  1.0, x1, y2,
       1.0, -1.0, x1, y1,
    ], dtype=np.float32)  # fmt: skip
    vertices_ptr = ffi.cast("float *", ffi.from_buffer(vertices))
    self.vbo = rl.rl_load_vertex_buffer(vertices_ptr, vertices.nbytes, False)
    rl.rl_enable_vertex_buffer(self.vbo)

    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint8)
    indices_ptr = ffi.cast("char *", ffi.from_buffer(indices))
    self.ibo = rl.rl_load_vertex_buffer_element(indices_ptr, indices.nbytes, False)
    rl.rl_enable_vertex_buffer_element(self.ibo)

    position_loc = rl.get_shader_location_attrib(self.program, "aPosition")
    rl.rl_enable_vertex_attribute(position_loc)
    rl.rl_set_vertex_attribute(position_loc, 2, RL_FLOAT, False, 4 * 4, 0)

    tex_coord_loc = rl.get_shader_location_attrib(self.program, "aTexCoord")
    rl.rl_enable_vertex_attribute(tex_coord_loc)
    rl.rl_set_vertex_attribute(tex_coord_loc, 2, RL_FLOAT, False, 4 * 4, 2 * 4)

    rl.rl_disable_vertex_array()

    with CheckGL("early init textures"):
      self.textures = (
        rl.rl_load_texture(None, 1, 1, PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAYSCALE, 1),
        rl.rl_load_texture(None, 1, 1, PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA, 1),
      )

    with CheckGL("set shader uniforms"):
      rl.rl_enable_shader(self.program.id)
      rl.rl_active_texture_slot(0)
      rl.rl_enable_texture(self.textures[0])
      rl.rl_set_uniform_sampler(rl.get_shader_location(self.program, "uTextureY"), ffi.cast("int", 0))
      rl.rl_active_texture_slot(1)
      rl.rl_enable_texture(self.textures[1])
      rl.rl_set_uniform_sampler(rl.get_shader_location(self.program, "uTextureUV"), ffi.cast("int", 1))

  def _start_vipc_thread(self) -> None:
    self.vipc_thread_stop_event.clear()
    self.vipc_thread = threading.Thread(target=self._vipc_thread_func, daemon=True)
    self.vipc_thread.start()

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
        if not self.vipc_client.connect(False):
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

    if self.textures:
      for tex_id in self.textures:
        rl.rl_unload_texture(tex_id)

    with CheckGL("_on_vipc_connected load textures"):
      self.textures = (
        rl.rl_load_texture(None, self.stream_width, self.stream_height, PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAYSCALE, 1),
        rl.rl_load_texture(None, self.stream_width // 2, self.stream_height // 2, PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA, 1),
      )

  def render(self) -> None:
    rl.clear_background(rl.BLACK)

    if self.vipc_connected_event.is_set():
      self.vipc_connected_event.clear()
      self._on_vipc_connected()

    if self.program is None or self.textures is None or self.vao is None or self.vbo is None or self.ibo is None:
      return

    with self.frame_lock:
      if not self.frames:
        return
      frame_id, frame = self.frames[-1]

    if self.prev_frame_id is not None:
      if frame_id == self.prev_frame_id:
        cloudlog.debug(f"Drawing same frame twice {frame_id}")
      elif frame_id != self.prev_frame_id + 1:
        cloudlog.debug(f"Skipped frame {frame_id}")
    self.prev_frame_id = frame_id

    rl.rl_viewport(0, 0, gui_app.width, gui_app.height)
    rl.rl_enable_vertex_array(self.vao)  # glBindVertexArray(vao)
    rl.rl_enable_shader(self.program.id)  # glUseProgram(id)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, c_int(1))

    with CheckGL("update textures"):
      buf_data = frame.data
      y_ptr = ffi.cast("unsigned char *", ffi.from_buffer(buf_data))
      uv_ptr = ffi.cast("unsigned char *", ffi.from_buffer(buf_data[frame.uv_offset:]))
      gl.glPixelStorei(gl.GL_UNPACK_ROW_LENGTH, c_int(self.stream_stride))
      rl.rl_update_texture(self.textures[0], 0, 0, self.stream_width, self.stream_height,
                           rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAYSCALE, y_ptr)
      gl.glPixelStorei(gl.GL_UNPACK_ROW_LENGTH, c_int(self.stream_stride // 2))
      rl.rl_update_texture(self.textures[1], 0, 0, self.stream_width // 2, self.stream_height // 2,
                           rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA, uv_ptr)

    rl.rl_enable_vertex_attribute(0)  # glEnableVertexAttribArray(0)

    with CheckGL("draw elements"):
      rl.rl_draw_vertex_array_elements(0, 6, None)

    rl.rl_disable_vertex_attribute(0)  # glDisableVertexAttribArray(0)
    rl.rl_disable_vertex_array()  # glBindVertexArray(0)

    rl.rl_disable_texture()  # glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, 0);
    # glActiveTexture(GL_TEXTURE0);
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, c_int(4))
    gl.glPixelStorei(gl.GL_UNPACK_ROW_LENGTH, c_int(0))

  def close(self) -> None:
    self.vipc_thread_stop_event.set()
    if self.vipc_thread and self.vipc_thread.is_alive():
      self.vipc_thread.join(timeout=2.0)
      if self.vipc_thread.is_alive():
        print(f"[{self.active_stream_type}] Thread did not exit cleanly")

    if self.program:
      rl.unload_shader(self.program)
      self.program = None

    if self.vao:
      rl.rl_unload_vertex_array(self.vao)
      self.vao = None

    if self.vbo:
      rl.rl_unload_vertex_buffer(self.vbo)
      self.vbo = None

    if self.ibo:
      rl.rl_unload_vertex_buffer(self.ibo)
      self.ibo = None

    if self.textures:
      for tex_id in self.textures:
        rl.rl_unload_texture(tex_id)
      self.textures = None

    gui_app.close()


if __name__ == "__main__":
  gui_app.init_window("watch3")
  road_camera_view = CameraView("camerad", VisionStreamType.VISION_STREAM_ROAD)
  for _ in gui_app.render():
    road_camera_view.render()
  road_camera_view.close()
  gui_app.close()
