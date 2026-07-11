import os
import queue
import subprocess
import threading

import ffmpeg as ffmpeg_pkg

from openpilot.cereal import log
from openpilot.common.swaglog import cloudlog
from openpilot.system.loggerd.encoder.encoder import V4L2_BUF_FLAG_KEYFRAME, EncoderInfo, FrameExtra, VideoEncoder, visionbuf_to_nv12

FFMPEG = os.path.join(ffmpeg_pkg.BIN_DIR, "ffmpeg")

DEBUG_ENCODER = int(os.getenv("DEBUG_ENCODER", "0"))


class H264Splitter:
  """Split a raw annexb H264 stream into access units. Requires the encoder to emit an AUD at the start of every AU."""
  AUD = b"\x00\x00\x01\x09"

  def __init__(self):
    self.buf = bytearray()
    self.scan_pos = 0

  def push(self, data: bytes) -> list[tuple[bytes, bool]]:
    self.buf += data
    packets = []
    scan = self.scan_pos
    while True:
      pos = self.buf.find(self.AUD, scan + 1)
      if pos < 0:
        break
      start = pos - 1 if self.buf[pos - 1] == 0 else pos  # 4-byte start code
      if start == 0:
        # the AUD opening the current access unit, not a boundary
        scan = pos + 1
        continue
      packets.append(self._packet(self.buf[:start]))
      del self.buf[:start]
      scan = 0
    self.scan_pos = max(0, len(self.buf) - len(self.AUD) - 1)
    return packets

  def flush(self) -> list[tuple[bytes, bool]]:
    packets = [self._packet(self.buf[:])] if len(self.buf) else []
    self.buf.clear()
    self.scan_pos = 0
    return packets

  @staticmethod
  def _packet(au: bytes) -> tuple[bytes, bool]:
    # keyframe if the access unit contains an IDR NAL
    keyframe = False
    pos = 0
    while (pos := au.find(b"\x00\x00\x01", pos)) >= 0:
      pos += 3
      if pos < len(au) and au[pos] & 0x1f == 5:
        keyframe = True
        break
    return bytes(au), keyframe


class MkvSplitter:
  """Extract SimpleBlock payloads from a streamed matroska file."""
  MASTER_IDS = (0x18538067, 0x1F43B675)  # Segment, Cluster
  SIMPLEBLOCK_ID = 0xA3

  def __init__(self):
    self.buf = bytearray()

  def push(self, data: bytes) -> list[tuple[bytes, bool]]:
    self.buf += data
    packets = []
    pos = 0
    while True:
      element = self._parse_element(pos)
      if element is None:
        break
      eid, size, hdr_len = element
      if eid in self.MASTER_IDS:
        pos += hdr_len  # descend into master elements
        continue
      if pos + hdr_len + size > len(self.buf):
        break
      if eid == self.SIMPLEBLOCK_ID:
        packets.append(self._block(self.buf[pos + hdr_len:pos + hdr_len + size]))
      pos += hdr_len + size
    del self.buf[:pos]
    return packets

  def flush(self) -> list[tuple[bytes, bool]]:
    self.buf.clear()
    return []

  def _parse_element(self, pos: int) -> tuple[int, int, int] | None:
    eid = self._read_vint(pos, strip_marker=False)
    if eid is None:
      return None
    size = self._read_vint(pos + eid[1])
    if size is None:
      return None
    return eid[0], size[0], eid[1] + size[1]

  def _read_vint(self, pos: int, strip_marker: bool = True) -> tuple[int, int] | None:
    if pos >= len(self.buf):
      return None
    first = self.buf[pos]
    if first == 0:
      raise ValueError("invalid EBML vint")
    length = 8 - first.bit_length() + 1
    if pos + length > len(self.buf):
      return None
    val = first & ((0x80 >> (length - 1)) - 1) if strip_marker else first
    for i in range(1, length):
      val = (val << 8) | self.buf[pos + i]
    return val, length

  @staticmethod
  def _block(block: bytes) -> tuple[bytes, bool]:
    # SimpleBlock: track number vint, s16 relative timestamp, u8 flags (0x80 = keyframe), frame data
    track_len = 8 - block[0].bit_length() + 1
    flags = block[track_len + 2]
    return bytes(block[track_len + 3:]), bool(flags & 0x80)


class FfmpegEncoder(VideoEncoder):
  def __init__(self, encoder_info: EncoderInfo, in_width: int, in_height: int):
    super().__init__(encoder_info, in_width, in_height)

    settings = encoder_info.get_settings(in_width)
    self.is_h264 = settings.encode_type == log.EncodeIndex.Type.qcameraH264

    cmd = [
      FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
      "-f", "rawvideo", "-pix_fmt", "nv12", "-video_size", f"{in_width}x{in_height}",
      "-framerate", str(encoder_info.fps), "-i", "pipe:0",
    ]
    if (self.out_width, self.out_height) != (in_width, in_height):
      cmd += ["-vf", f"scale={self.out_width}:{self.out_height}"]
    cmd += ["-pix_fmt", "yuv420p"]
    if self.is_h264:
      # veryfast keeps the encoder delay and CPU usage low
      cmd += ["-c:v", "libx264", "-preset", "veryfast", "-b:v", str(settings.bitrate), "-g", str(settings.gop_size), "-x264-params", "aud=1",
              "-f", "rawvideo"]
    else:
      cmd += ["-c:v", "ffvhuff",
              "-f", "matroska", "-cluster_time_limit", "0", "-cluster_size_limit", "0", "-write_crc32", "0", "-live", "1"]
    cmd += ["-flush_packets", "1", "pipe:1"]
    self.cmd = cmd

    self.proc: subprocess.Popen | None = None
    self.packets: queue.Queue[tuple[bytes, bool]] = queue.Queue()
    # a few frames of buffer to absorb encoder startup, then block for backpressure like the libavcodec implementation
    self.frames: queue.Queue[bytes | None] = queue.Queue(maxsize=4)
    self.is_open = False
    self.segment_num = -1
    self.counter = 0

  def _read_handler(self, proc: subprocess.Popen) -> None:
    splitter = H264Splitter() if self.is_h264 else MkvSplitter()
    while True:
      data = proc.stdout.read1(1 << 20)
      if not data:
        break
      for packet in splitter.push(data):
        self.packets.put(packet)
    for packet in splitter.flush():
      self.packets.put(packet)

  def _write_handler(self, proc: subprocess.Popen) -> None:
    # the encoder can block its stdin pipe while starting up, keep writes off the encode_frame path
    while (frame := self.frames.get()) is not None:
      try:
        proc.stdin.write(frame)
        proc.stdin.flush()
      except (BrokenPipeError, ValueError):
        break

  def encoder_open(self) -> None:
    self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    self.read_thread = threading.Thread(target=self._read_handler, args=(self.proc,), name=f"read-{self.encoder_info.publish_name}")
    self.read_thread.start()
    self.write_thread = threading.Thread(target=self._write_handler, args=(self.proc,), name=f"write-{self.encoder_info.publish_name}")
    self.write_thread.start()
    self.is_open = True
    self.segment_num += 1
    self.counter = 0

  def encoder_close(self) -> None:
    if not self.is_open:
      return

    # like the previous libavcodec implementation, frames still buffered in the encoder are dropped
    self.proc.kill()
    try:
      self.frames.put_nowait(None)  # wake up the write thread if it's idle; a killed pipe wakes it up otherwise
    except queue.Full:
      pass
    try:
      self.proc.stdin.close()
    except BrokenPipeError:
      pass
    self.proc.wait()
    self.write_thread.join()
    self.read_thread.join()
    self.proc.stdout.close()
    self.proc = None
    for q in (self.packets, self.frames):
      while not q.empty():
        q.get_nowait()
    self.is_open = False

  def set_bitrate(self, bitrate: int) -> None:
    cloudlog.error(f"adaptive bitrate is not supported for ffmpeg encoder {self.encoder_info.publish_name}")

  def request_keyframe(self) -> None:
    cloudlog.error(f"keyframe request is not supported for ffmpeg encoder {self.encoder_info.publish_name}")

  def encode_frame(self, buf, extra: FrameExtra) -> int:
    assert buf.width == self.in_width
    assert buf.height == self.in_height

    ret = self.counter
    if self.proc.poll() is None:
      self.frames.put(bytes(visionbuf_to_nv12(buf)))
    else:
      cloudlog.error(f"ffmpeg encoder {self.encoder_info.publish_name} died")
      ret = -1

    while not self.packets.empty():
      dat, keyframe = self.packets.get_nowait()

      if DEBUG_ENCODER:
        print(f"{self.encoder_info.publish_name:>20} got {len(dat):8d} bytes keyframe {keyframe:d} idx {self.counter:4d} id {extra.frame_id:8d}")

      self.publisher_publish(self.segment_num, self.counter, extra,
                             V4L2_BUF_FLAG_KEYFRAME if keyframe else 0, b"", dat)
      self.counter += 1

    return ret
