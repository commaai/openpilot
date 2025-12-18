import os
import subprocess
import threading
import time
import numpy as np
import pyray as rl
from openpilot.system.ui.widgets import Widget


class VideoPlayer(Widget):
  def __init__(self, video_path: str):
    super().__init__()
    self.video_path = video_path
    self.texture: rl.Texture | None = None
    self.ffmpeg_proc: subprocess.Popen | None = None
    self.current_frame: np.ndarray | None = None
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

    # Try hardware acceleration first, fallback to software
    cmd = [
      'ffmpeg', '-hwaccel', 'auto',  # Try hardware acceleration
      '-i', self.video_path,
      '-f', 'rawvideo',
      '-pix_fmt', 'rgb24',
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
        bufsize=self.frame_width * self.frame_height * 3 * 2  # Larger buffer
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

    # Skip frames if we're behind to catch up
    if elapsed < self.frame_duration:
      return

    # Calculate how many frames we should skip if behind
    frames_to_skip = int(elapsed / self.frame_duration) - 1
    frames_to_skip = min(frames_to_skip, 5)  # Max skip 5 frames at once

    frame_size = self.frame_width * self.frame_height * 3
    try:
      # Skip frames if we're behind
      for _ in range(frames_to_skip):
        self.ffmpeg_proc.stdout.read(frame_size)

      frame_data = self.ffmpeg_proc.stdout.read(frame_size)
      if len(frame_data) == frame_size:
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
          self.frame_height, self.frame_width, 3
        )
        # Flip vertically (OpenGL convention) - use view instead of copy when possible
        frame = np.ascontiguousarray(np.flipud(frame))

        with self.frame_lock:
          self.current_frame = frame
          # Pre-convert to RGBA for faster rendering
          if self.rgba_frame is None:
            self.rgba_frame = np.zeros((self.frame_height, self.frame_width, 4), dtype=np.uint8)
          self.rgba_frame[:, :, :3] = frame
          self.rgba_frame[:, :, 3] = 255  # Alpha channel
        self.last_frame_time = current_time
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
    with self.frame_lock:
      if self.rgba_frame is not None and self.texture is not None:
        # Update texture with pre-converted RGBA frame data
        rl.update_texture(self.texture, rl.ffi.cast("void *", self.rgba_frame.ctypes.data))

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
        rl.draw_texture_pro(self.texture, src_rect, dst_rect, rl.Vector2(0, 0), 0.0, rl.WHITE)
      else:
        # Draw loading/black screen
        rl.draw_rectangle_rec(rect, rl.BLACK)

  def __del__(self):
    """Cleanup"""
    if self.ffmpeg_proc is not None:
      self.ffmpeg_proc.terminate()
      self.ffmpeg_proc.wait()
      self.ffmpeg_proc = None
    if self.texture is not None:
      rl.unload_texture(self.texture)

