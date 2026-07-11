import os
import subprocess
import threading

import ffmpeg as ffmpeg_pkg

from openpilot.cereal import messaging
from openpilot.common.swaglog import cloudlog
from openpilot.system.loggerd.encoder.encoder import FrameExtra, visionbuf_to_nv12

FFMPEG = os.path.join(ffmpeg_pkg.BIN_DIR, "ffmpeg")

# Lower qscale = higher quality / bigger files for MJPEG.
MJPEG_QSCALE = 7


class JpegEncoder:
  def __init__(self, publish_name: str, width: int, height: int):
    self.publish_name = publish_name
    self.thumbnail_width = width
    self.thumbnail_height = height
    self.pm = messaging.PubMaster([publish_name])
    self.thread: threading.Thread | None = None

  def push_thumbnail(self, buf, extra: FrameExtra) -> None:
    # spawning ffmpeg takes long enough to lag the encoder, keep it off the hot loop
    if self.thread is not None and self.thread.is_alive():
      cloudlog.error("skipping thumbnail, encoder is still running")
      return
    nv12 = bytes(visionbuf_to_nv12(buf))  # copy out of the VisionIPC buffer
    self.thread = threading.Thread(target=self._generate_thumbnail, args=(nv12, buf.width, buf.height, extra), name="thumbnail")
    self.thread.start()

  def _generate_thumbnail(self, nv12: bytes, width: int, height: int, extra: FrameExtra) -> None:
    cmd = [
      FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
      "-f", "rawvideo", "-pix_fmt", "nv12", "-video_size", f"{width}x{height}", "-i", "pipe:0",
      "-vf", f"scale={self.thumbnail_width}:{self.thumbnail_height}",
      "-c:v", "mjpeg", "-pix_fmt", "yuvj420p", "-q:v", str(MJPEG_QSCALE), "-frames:v", "1",
      "-f", "rawvideo", "pipe:1",
    ]
    try:
      jpeg = subprocess.run(cmd, input=nv12, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
      cloudlog.error(f"thumbnail ffmpeg error: {e.stderr}")
      jpeg = b""

    msg = messaging.new_message('thumbnail', valid=True)
    msg.thumbnail.frameId = extra.frame_id
    msg.thumbnail.timestampEof = extra.timestamp_eof
    msg.thumbnail.thumbnail = jpeg
    self.pm.send(self.publish_name, msg)
