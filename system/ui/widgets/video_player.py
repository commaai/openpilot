import os
import subprocess
import threading
import time
import numpy as np
import pyray as rl
from openpilot.system.ui.widgets import Widget

# Benchmarking
_benchmark_enabled = True
_benchmark_times = {
  'frame_read': [],
  'rgb_to_rgba': [],
  'texture_update': [],
  'render': [],
  'total_frame': []
}


class VideoPlayer(Widget):
  def __init__(self, video_path: str):
    super().__init__()
    self.video_path = video_path
    self.texture: rl.Texture | None = None
    self.ffmpeg_proc: subprocess.Popen | None = None
    self.current_frame: np.ndarray | None = None
    self.rgba_frame: np.ndarray | None = None  # Pre-allocated RGBA buffer
    self.frame_width = 0
    self.frame_height = 0
    self.playing = False
    self.start_time = 0.0
    self.last_frame_time = 0.0
    self.frame_lock = threading.Lock()
    self.frame_duration = 1.0 / 30.0  # Default 30 fps

  def _load_video(self):
    """Load video and get dimensions"""
    if not os.path.exists(self.video_path):
      print(f"Video file not found: {self.video_path}")
      return False

    # Get video info
    probe_cmd = [
      'ffprobe', '-v', 'error', '-select_streams', 'v:0',
      '-show_entries', 'stream=width,height,r_frame_rate',
      '-of', 'csv=p=0', self.video_path
    ]
    try:
      result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
      info = result.stdout.strip().split(',')
      self.frame_width = int(info[0])
      self.frame_height = int(info[1])
      fps_str = info[2]
      # Parse fps (e.g., "30/1" -> 30.0)
      if '/' in fps_str:
        num, den = map(int, fps_str.split('/'))
        self.fps = num / den if den > 0 else 30.0
      else:
        self.fps = float(fps_str) if fps_str else 30.0

      self.frame_duration = 1.0 / self.fps
      print(f"Video info: {self.frame_width}x{self.frame_height} @ {self.fps} fps")
      return True
    except Exception as e:
      print(f"Failed to probe video: {e}")
      return False

  def _start_ffmpeg(self):
    """Start ffmpeg process to decode video frames"""
    if self.ffmpeg_proc is not None:
      return

    # Use RGBA output directly to avoid conversion
    cmd = [
      'ffmpeg', '-hwaccel', 'auto',  # Try hardware acceleration
      '-i', self.video_path,
      '-f', 'rawvideo',
      '-pix_fmt', 'rgba',  # Output RGBA directly
      '-vcodec', 'rawvideo',
      '-an', '-sn',  # No audio, no subtitles
      '-threads', '2',  # Limit threads for better performance
      '-'
    ]

    try:
      self.ffmpeg_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=self.frame_width * self.frame_height * 4 * 2  # RGBA = 4 bytes per pixel
      )
      self.playing = True
      self.start_time = time.monotonic()
      self.last_frame_time = self.start_time
      print("FFmpeg started")
    except Exception as e:
      print(f"Failed to start ffmpeg: {e}")
      self.ffmpeg_proc = None

  def _update_frame(self):
    """Read next frame from ffmpeg based on timing"""
    if self.ffmpeg_proc is None or self.ffmpeg_proc.stdout is None:
      return

    current_time = time.monotonic()
    elapsed = current_time - self.last_frame_time

    # Only read new frame if enough time has passed
    if elapsed < self.frame_duration:
      return

    frame_start = time.monotonic()
    frame_size = self.frame_width * self.frame_height * 4  # RGBA = 4 bytes per pixel

    try:
      read_start = time.monotonic()
      # Pre-allocate buffer if needed
      with self.frame_lock:
        if self.rgba_frame is None:
          self.rgba_frame = np.zeros((self.frame_height, self.frame_width, 4), dtype=np.uint8)
        rgba_buffer = self.rgba_frame

      # Read directly into numpy array buffer using memoryview - avoids intermediate copy!
      frame_bytes = self.ffmpeg_proc.stdout.read(frame_size)
      read_time = time.monotonic() - read_start
      if _benchmark_enabled and len(_benchmark_times['frame_read']) < 1000:
        _benchmark_times['frame_read'].append(read_time)

      if len(frame_bytes) == frame_size:
        rgba_start = time.monotonic()
        # Read directly into buffer - much faster than copy!
        rgba_view = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
          self.frame_height, self.frame_width, 4
        )
        # Copy directly into pre-allocated buffer
        with self.frame_lock:
          np.copyto(rgba_buffer, rgba_view)
          self.current_frame = rgba_buffer[:, :, :3]  # RGB slice for compatibility
        rgba_time = time.monotonic() - rgba_start
        if _benchmark_enabled and len(_benchmark_times['rgb_to_rgba']) < 1000:
          _benchmark_times['rgb_to_rgba'].append(rgba_time)

        self.last_frame_time = current_time
        total_frame_time = time.monotonic() - frame_start
        if _benchmark_enabled and len(_benchmark_times['total_frame']) < 1000:
          _benchmark_times['total_frame'].append(total_frame_time)
      else:
        # End of video, restart
        print("Video ended, restarting...")
        self.ffmpeg_proc.terminate()
        self.ffmpeg_proc.wait()
        self.ffmpeg_proc = None
        self._start_ffmpeg()
    except Exception as e:
      print(f"Error reading frame: {e}")

  def _render(self, rect: rl.Rectangle):
    if not self.playing:
      if self._load_video():
        self._start_ffmpeg()
        if self.texture is None and self.frame_width > 0:
          # Create texture
          temp_image = rl.gen_image_color(self.frame_width, self.frame_height, rl.BLACK)
          self.texture = rl.load_texture_from_image(temp_image)
          rl.unload_image(temp_image)
      else:
        # Draw placeholder
        rl.draw_rectangle_rec(rect, rl.DARKGRAY)
        rl.draw_text("Video not found", int(rect.x + 10), int(rect.y + 10), 20, rl.WHITE)
        return

    # Update frame based on timing
    if self.ffmpeg_proc is not None:
      self._update_frame()

    # Render current frame
    render_start = time.monotonic()
    rgba_frame_copy = None
    texture_ref = None
    with self.frame_lock:
      if self.rgba_frame is not None and self.texture is not None:
        # Copy reference to avoid holding lock during texture update
        rgba_frame_copy = self.rgba_frame
        texture_ref = self.texture

    if rgba_frame_copy is not None and texture_ref is not None:
      # Update texture outside lock to reduce contention
      texture_start = time.monotonic()
      rl.update_texture(texture_ref, rl.ffi.cast("void *", rgba_frame_copy.ctypes.data))
      texture_time = time.monotonic() - texture_start
      if _benchmark_enabled and len(_benchmark_times['texture_update']) < 1000:
        _benchmark_times['texture_update'].append(texture_time)

      # Calculate aspect ratio preserving rect
      video_aspect = self.frame_width / self.frame_height
      rect_aspect = rect.width / rect.height

      if video_aspect > rect_aspect:
        # Video is wider, fit to width
        draw_height = rect.width / video_aspect
        draw_y = rect.y + (rect.height - draw_height) / 2
        dst_rect = rl.Rectangle(rect.x, draw_y, rect.width, draw_height)
      else:
        # Video is taller, fit to height
        draw_width = rect.height * video_aspect
        draw_x = rect.x + (rect.width - draw_width) / 2
        dst_rect = rl.Rectangle(draw_x, rect.y, draw_width, rect.height)

      src_rect = rl.Rectangle(0, 0, self.frame_width, -self.frame_height)
      rl.draw_texture_pro(texture_ref, src_rect, dst_rect, rl.Vector2(0, 0), 0.0, rl.WHITE)

      render_time = time.monotonic() - render_start
      if _benchmark_enabled and len(_benchmark_times['render']) < 1000:
        _benchmark_times['render'].append(render_time)

        # Print stats every 60 frames
        if len(_benchmark_times['render']) % 60 == 0:
          self._print_benchmark_stats()
    else:
      # Draw loading/black screen
      rl.draw_rectangle_rec(rect, rl.BLACK)

  def _print_benchmark_stats(self):
    """Print benchmark statistics"""
    if not _benchmark_enabled:
      return

    print("\n=== Benchmark Stats (last 60 frames) ===")
    for key, times in _benchmark_times.items():
      if times:
        recent = times[-60:]  # Last 60 frames
        avg = sum(recent) / len(recent) * 1000  # Convert to ms
        max_time = max(recent) * 1000
        min_time = min(recent) * 1000
        print(f"{key:20s}: avg={avg:6.2f}ms, min={min_time:6.2f}ms, max={max_time:6.2f}ms")
    print("=" * 50)

    # Clear old data to prevent memory growth
    for key in _benchmark_times:
      if len(_benchmark_times[key]) > 120:
        _benchmark_times[key] = _benchmark_times[key][-60:]

  def __del__(self):
    """Cleanup"""
    if self.ffmpeg_proc is not None:
      self.ffmpeg_proc.terminate()
      self.ffmpeg_proc.wait()
      self.ffmpeg_proc = None
    if self.texture is not None:
      rl.unload_texture(self.texture)
    # Print final stats
    if _benchmark_enabled:
      self._print_benchmark_stats()

