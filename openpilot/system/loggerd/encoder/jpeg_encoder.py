import os
import subprocess
import threading

import ffmpeg as ffmpeg_pkg

from openpilot.cereal import messaging
from openpilot.common.swaglog import cloudlog
from openpilot.system.loggerd.encoder.encoder import FrameExtra, drop_realtime_in_child

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
    # spawning ffmpeg takes long enough to lag the encoder, keep everything but one buffer copy off the hot path
    if self.thread is not None and self.thread.is_alive():
      cloudlog.error("skipping thumbnail, encoder is still running")
      return
    padded_height = buf.uv_offset // buf.stride
    nv12 = bytes(buf.data[:buf.stride * padded_height * 3 // 2])
    self.thread = threading.Thread(target=self._generate_thumbnail, name="thumbnail",
                                   args=(nv12, buf.width, buf.height, buf.stride, padded_height, extra))
    self.thread.start()

  def _generate_thumbnail(self, nv12: bytes, width: int, height: int, stride: int, padded_height: int, extra: FrameExtra) -> None:
    cmd = [
      FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
      "-f", "rawvideo", "-pix_fmt", "nv12", "-video_size", f"{stride}x{padded_height}", "-i", "pipe:0",
      "-vf", f"crop={width}:{height}:0:0,scale={self.thumbnail_width}:{self.thumbnail_height}",
      "-c:v", "mjpeg", "-pix_fmt", "yuvj420p", "-q:v", str(MJPEG_QSCALE), "-frames:v", "1",
      "-f", "rawvideo", "pipe:1",
    ]
    try:
      jpeg = subprocess.run(cmd, input=nv12, capture_output=True, check=True, preexec_fn=drop_realtime_in_child).stdout
    except subprocess.CalledProcessError as e:
      cloudlog.error(f"thumbnail ffmpeg error: {e.stderr}")
      jpeg = b""

    msg = messaging.new_message('thumbnail', valid=True)
    msg.thumbnail.frameId = extra.frame_id
    msg.thumbnail.timestampEof = extra.timestamp_eof
    msg.thumbnail.thumbnail = jpeg
    self.pm.send(self.publish_name, msg)
