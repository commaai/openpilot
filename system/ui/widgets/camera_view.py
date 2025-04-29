import threading
from collections import deque
from enum import IntEnum
import pyray as rl
import cereal.messaging as messaging
from openpilot.system.ui.lib.application import gui_app


# Match VisionStreamType from C++ headers
class VisionStreamType(IntEnum):
  VISION_STREAM_ROAD = 0
  VISION_STREAM_WIDE_ROAD = 1
  VISION_STREAM_DRIVER = 2
  VISION_STREAM_MAX = 3


# Match cereal event names
STREAM_TO_EVENT = {
  VisionStreamType.VISION_STREAM_ROAD: "roadCameraState",
  VisionStreamType.VISION_STREAM_WIDE_ROAD: "wideRoadCameraState",
  VisionStreamType.VISION_STREAM_DRIVER: "driverCameraState",
}

FRAME_BUFFER_SIZE = 5  # Number of frames to buffer

# Basic GLSL shaders for YUV->RGB conversion (NV12/NV21 specific)
# Note: pyray might need specific shader version headers depending on the platform (#version 330 or #version 300 es)
# Check Raylib examples for precise shader requirements. These are conceptual.

VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec2 vertexTexCoord;
out vec2 fragTexCoord;
uniform mat4 mvp; // Model-View-Projection matrix

void main()
{
    fragTexCoord = vertexTexCoord;
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""

# NV12 Fragment Shader (UV plane is interleaved U/V)
FRAGMENT_SHADER_NV12 = """
#version 330 core
in vec2 fragTexCoord;
out vec4 finalColor;

uniform sampler2D texY;
uniform sampler2D texUV;
uniform vec2 frameSize; // Width, Height of the Y plane

mat3 colorConversionMatrix = mat3(
    1.0, 1.0, 1.0,
    0.0, -0.39465, 2.03211,
    1.13983, -0.58060, 0.0
);

void main()
{
    // Y plane (luminance)
    float y = texture(texY, fragTexCoord).r;

    // UV plane (chrominance), coordinates are half resolution
    // Fetch the interleaved UV pair
    vec2 uv = texture(texUV, fragTexCoord).ra; // Use .ra for GL_RG format

    // Center the UV values around 0
    uv -= vec2(0.5, 0.5);

    // Convert YUV to RGB
    vec3 yuv = vec3(y, uv.x, uv.y);
    vec3 rgb = colorConversionMatrix * yuv;

    finalColor = vec4(rgb, 1.0);

    // Optional: Apply gamma correction like in Qt version
    // finalColor.rgb = pow(finalColor.rgb, vec3(1.0/1.28));
}
"""


class RaylibCameraView:
  def __init__(self, stream_type: VisionStreamType, initial_width=0, initial_height=0):
    self.stream_type = stream_type
    self.event_name = STREAM_TO_EVENT[self.stream_type]

    # Frame data
    self.frames: deque[tuple[int, bytes, int, int, int]] = deque(maxlen=FRAME_BUFFER_SIZE)  # frame_id, image_bytes, w, h, stride
    self.latest_frame_id = -1
    self.stream_width = initial_width
    self.stream_height = initial_height
    self.stream_stride = 0  # Will be updated when first frame arrives

    # Threading
    self.frame_lock = threading.Lock()
    self.frame_thread = threading.Thread(target=self._frame_fetch_thread)
    self.frame_thread.daemon = True
    self._stop_event = threading.Event()

    # Rendering resources (initialized in _init_rendering)
    self.tex_y: rl.Texture | None = None
    self.tex_uv: rl.Texture | None = None
    self.shader: rl.Shader | None = None
    self._rendering_inited = False

    # Messaging
    self.sm: messaging.SubMaster | None = None

    # Start frame fetching
    self.frame_thread.start()

  def _init_rendering(self):
    if not rl.is_window_ready():
      print("Warning: Raylib window not ready for rendering init")
      return

    # Load shader
    self.shader = rl.load_shader_from_memory(VERTEX_SHADER, FRAGMENT_SHADER_NV12)  # Assuming NV12 for now
    if self.shader.id <= 0:
      print("Error: Failed to load camera view shader")
      # Handle error appropriately
      return

    # Get shader uniform locations
    # self.shader_frame_size_loc = rl.get_shader_location(self.shader, "frameSize")

    # Create placeholder textures
    # We create them here but load data during render()
    # Texture formats: Y=Grayscale, UV=RG (2 channels)
    # Need to make sure PIXELFORMAT matches the data layout and shader expectations
    img_y = rl.gen_image_color(self.stream_width, self.stream_height, rl.BLANK)
    self.tex_y = rl.load_texture_from_image(img_y)
    rl.set_texture_filter(self.tex_y, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
    rl.unload_image(img_y)

    img_uv = rl.gen_image_color(self.stream_width // 2, self.stream_height // 2, rl.BLANK)
    self.tex_uv = rl.load_texture_from_image(img_uv)
    rl.set_texture_filter(self.tex_uv, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
    rl.unload_image(img_uv)

    # Texture slots (match shader uniforms)
    rl.set_shader_value_texture(self.shader, rl.get_shader_location(self.shader, "texY"), self.tex_y)
    rl.set_shader_value_texture(self.shader, rl.get_shader_location(self.shader, "texUV"), self.tex_uv)

    self._rendering_inited = True
    print(f"CameraView rendering initialized for {self.event_name}")

  def _frame_fetch_thread(self):
    """Dedicated thread to receive frames and put them in the buffer."""
    self.sm = messaging.SubMaster([self.event_name])
    print(f"Frame fetch thread started for {self.event_name}")
    while not self._stop_event.is_set():
      self.sm.update()
      if self.sm.updated[self.event_name]:
        msg = self.sm[self.event_name]
        img_bytes = msg.image  # Assuming this is the raw YUV byte buffer
        frame_id = msg.frameId

        # Update stream dimensions if necessary (ideally happens only once)
        new_width, new_height, new_stride = msg.width, msg.height, msg.stride
        if new_width != self.stream_width or new_height != self.stream_height:
          print(f"Stream dimensions changed for {self.event_name}: {new_width}x{new_height}, stride {new_stride}")
          # TODO: Need to handle texture recreation or update if size changes.
          # This might require signalling the main thread or careful locking.
          # For now, we assume dimensions are constant after the first frame.
          if not self._rendering_inited:
            self.stream_width = new_width
            self.stream_height = new_height
            self.stream_stride = new_stride
            # Attempt init again if dimensions are now valid
            if self.stream_width > 0 and self.stream_height > 0:
              # Need to schedule this on the main thread
              # For now, we rely on render() checking _rendering_inited
              pass

        with self.frame_lock:
          self.frames.append((frame_id, img_bytes, new_width, new_height, new_stride))
          self.latest_frame_id = frame_id
          # Update dimensions here too, protected by lock
          self.stream_width = new_width
          self.stream_height = new_height
          self.stream_stride = new_stride

    print(f"Frame fetch thread stopped for {self.event_name}")

  def update_frame_data(self, frame_data: bytes, width: int, height: int, stride: int):
    """Update Y and UV textures with new frame data."""
    if not self._rendering_inited or self.tex_y is None or self.tex_uv is None:
      return

    # Assuming NV12 format (Y plane followed by interleaved UV plane)
    y_plane_size = stride * height
    uv_plane_offset = y_plane_size  # NV12 specific

    # Note: Need to handle stride correctly. The actual data width might be less than stride.
    # Using UpdateTextureRec might be necessary if UpdateTexture doesn't handle stride.
    # For simplicity here, we assume UpdateTexture works okay if data buffer matches WxH.
    # A CFFI helper might be better for direct memory access with stride.

    # Update Y texture
    # Need a pointer/memoryview to the Y part of frame_data
    # pyray's UpdateTexture expects ctypes data or sequence supporting buffer protocol
    y_data = frame_data[:y_plane_size]
    if len(y_data) == width * height:  # Basic check
      # rl.update_texture(self.tex_y, y_data) # Need correct format conversion/pointer
      # Placeholder: using update_texture_rec might be needed for stride
      # For now, this part is conceptual until pyray specifics are nailed down
      pass
    else:
      print(f"Warning: Y data size mismatch {len(y_data)} vs {width * height}")

    # Update UV texture (needs de-interleaving or specific shader handling)
    uv_data = frame_data[uv_plane_offset:]
    if len(uv_data) == (width // 2) * (height // 2) * 2:  # NV12 UV plane size
      # Again, UpdateTexture needs correct format (likely PIXELFORMAT_UNCOMPRESSED_R8G8)
      # rl.update_texture(self.tex_uv, uv_data) # Needs format/pointer
      pass
    else:
      print(f"Warning: UV data size mismatch {len(uv_data)} vs {(width // 2) * (height // 2) * 2}")

  def render(self, dest_rect: rl.Rectangle):
    """Draw the latest camera frame within the specified rectangle."""
    # Ensure rendering resources are initialized (needs valid stream dimensions)
    if not self._rendering_inited and self.stream_width > 0 and self.stream_height > 0:
      self._init_rendering()

    if not self._rendering_inited or self.shader is None:
      # Optionally draw a placeholder/loading indicator
      rl.draw_rectangle_rec(dest_rect, rl.DARKGRAY)
      rl.draw_text_ex(gui_app.font(), "Waiting for camera...", rl.Vector2(dest_rect.x + 20, dest_rect.y + 20), 40, 0, rl.WHITE)
      return

    frame_to_draw = None
    with self.frame_lock:
      if self.frames:
        # Draw the latest frame
        frame_to_draw = self.frames[-1]

    if frame_to_draw:
      frame_id, img_bytes, width, height, stride = frame_to_draw
      # This is the critical part: update textures with the frame data
      self.update_frame_data(img_bytes, width, height, stride)

      # Setup shader and draw
      rl.begin_shader_mode(self.shader)
      # Update uniforms if needed (e.g., frameSize)
      # rl.set_shader_value(self.shader, self.shader_frame_size_loc, (float(width), float(height)), rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2)

      # Calculate aspect ratio preserving destination rect within dest_rect
      # TODO: Implement calc_frame_matrix equivalent logic here using Raylib matrix functions
      # or by setting the source/dest rectangles in DrawTexturePro correctly.
      # For now, just stretch to fill dest_rect.
      source_rec = rl.Rectangle(0, 0, float(self.stream_width), float(self.stream_height))
      origin = rl.Vector2(0, 0)
      rotation = 0.0
      tint = rl.WHITE

      # We draw the Y texture, the shader samples Y and UV textures to produce RGB
      # The coordinate system for drawing might need adjustment.
      rl.draw_texture_pro(self.tex_y, source_rec, dest_rect, origin, rotation, tint)

      rl.end_shader_mode()
    else:
      # Draw placeholder if no frames yet
      rl.draw_rectangle_rec(dest_rect, rl.DARKGRAY)
      rl.draw_text("No camera frames", int(dest_rect.x + 20), int(dest_rect.y + 20), 40, rl.WHITE)

  def set_stream_type(self, stream_type: VisionStreamType):
    if self.stream_type != stream_type:
      print(f"Switching camera view from {self.stream_type} to {stream_type}")
      # Signal the thread to change subscription (needs mechanism)
      # For now, requires recreating the object or restarting the thread cleanly.
      # Simplified: Stop old thread, start new one (requires care with locks/resources)
      self.stop()
      self.stream_type = stream_type
      self.event_name = STREAM_TO_EVENT[self.stream_type]
      # Reset state
      with self.frame_lock:
        self.frames.clear()
        self.latest_frame_id = -1
        self._rendering_inited = False  # Need to re-init textures/shader potentially
        if self.tex_y:
          rl.unload_texture(self.tex_y)
        if self.tex_uv:
          rl.unload_texture(self.tex_uv)
        if self.shader:
          rl.unload_shader(self.shader)
        self.tex_y, self.tex_uv, self.shader = None, None, None

      # Restart thread
      self._stop_event.clear()
      self.frame_thread = threading.Thread(target=self._frame_fetch_thread)
      self.frame_thread.daemon = True
      self.frame_thread.start()

  def stop(self):
    """Stop the frame fetching thread."""
    self._stop_event.set()
    if self.frame_thread.is_alive():
      self.frame_thread.join(timeout=2.0)  # Wait briefly
      if self.frame_thread.is_alive():
        print(f"Warning: Frame fetch thread for {self.event_name} did not join cleanly")

    # Consider unloading textures/shader here if appropriate in app lifecycle
    # if self._rendering_inited:
    #     if self.tex_y: rl.unload_texture(self.tex_y)
    #     if self.tex_uv: rl.unload_texture(self.tex_uv)
    #     if self.shader: rl.unload_shader(self.shader)
    #     self.tex_y, self.tex_uv, self.shader = None, None, None
    #     self._rendering_inited = False

  def __del__(self):
    self.stop()


if __name__ == "__main__":
  gui_app.init_window("CameraView")
  cv = RaylibCameraView(VisionStreamType.VISION_STREAM_ROAD)
  rect = rl.Rectangle(0, 0, gui_app.width, gui_app.height)
  for _ in gui_app.render():
    cv.render(rect)
  cv.stop()
  gui_app.close_window()
