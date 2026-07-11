import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from openpilot.cereal import log, messaging
from openpilot.common.hardware import PC

MAIN_FPS = 20
MAIN_ENCODE_TYPE = log.EncodeIndex.Type.bigBoxLossless if PC else log.EncodeIndex.Type.fullHEVC

V4L2_BUF_FLAG_KEYFRAME = 0x8


class FrameExtra(NamedTuple):
  frame_id: int
  timestamp_sof: int
  timestamp_eof: int


@dataclass
class EncoderSettings:
  encode_type: log.EncodeIndex.Type
  bitrate: int
  gop_size: int
  b_frames: int = 0  # we don't use b frames

  @staticmethod
  def main_encoder_settings(in_width: int) -> 'EncoderSettings':
    if in_width <= 1344:
      return EncoderSettings(encode_type=MAIN_ENCODE_TYPE, bitrate=5_000_000, gop_size=20)
    else:
      return EncoderSettings(encode_type=MAIN_ENCODE_TYPE, bitrate=10_000_000, gop_size=30)

  @staticmethod
  def qcam_encoder_settings(in_width: int) -> 'EncoderSettings':
    return EncoderSettings(encode_type=log.EncodeIndex.Type.qcameraH264, bitrate=256_000, gop_size=15)

  @staticmethod
  def stream_encoder_settings(in_width: int) -> 'EncoderSettings':
    bitrate = int(os.getenv("STREAM_BITRATE", "5000000"))
    return EncoderSettings(encode_type=log.EncodeIndex.Type.qcameraH264, bitrate=bitrate, gop_size=5)


@dataclass
class EncoderInfo:
  publish_name: str
  get_settings: Callable[[int], EncoderSettings]
  thumbnail_name: str | None = None
  is_live: bool = False
  frame_width: int = -1
  frame_height: int = -1
  fps: int = MAIN_FPS


@dataclass
class LogCameraInfo:
  thread_name: str
  stream_type: int
  encoder_infos: list[EncoderInfo] = field(default_factory=list)
  fps: int = MAIN_FPS


class VideoEncoder:
  def __init__(self, encoder_info: EncoderInfo, in_width: int, in_height: int):
    self.encoder_info = encoder_info
    self.in_width = in_width
    self.in_height = in_height
    self.out_width = encoder_info.frame_width if encoder_info.frame_width > 0 else in_width
    self.out_height = encoder_info.frame_height if encoder_info.frame_height > 0 else in_height

    self.cnt = 0  # total frames encoded
    self.pm = messaging.PubMaster([encoder_info.publish_name])

  def encode_frame(self, buf, extra: FrameExtra) -> int:
    raise NotImplementedError

  def encoder_open(self) -> None:
    raise NotImplementedError

  def encoder_close(self) -> None:
    raise NotImplementedError

  def close(self) -> None:
    self.encoder_close()

  def set_bitrate(self, bitrate: int) -> None:
    raise NotImplementedError

  def request_keyframe(self) -> None:
    raise NotImplementedError

  def publisher_publish(self, segment_num: int, idx: int, extra: FrameExtra, flags: int, header: bytes, dat: bytes) -> None:
    msg = messaging.new_message(self.encoder_info.publish_name, valid=True)
    edat = getattr(msg, self.encoder_info.publish_name)
    edat.unixTimestampNanos = time.time_ns()
    edata = edat.idx
    edata.frameId = extra.frame_id
    edata.timestampSof = extra.timestamp_sof
    edata.timestampEof = extra.timestamp_eof
    edata.type = self.encoder_info.get_settings(self.in_width).encode_type
    edata.encodeId = self.cnt
    edata.segmentNum = segment_num
    edata.segmentId = idx
    edata.flags = flags
    edata.len = len(dat)
    edat.data = dat
    edat.width = self.out_width
    edat.height = self.out_height
    if flags & V4L2_BUF_FLAG_KEYFRAME:
      edat.header = header

    self.pm.send(self.encoder_info.publish_name, msg)
    self.cnt += 1


def visionbuf_to_nv12(buf) -> bytes | memoryview:
  """Strip the stride padding from a VisionBuf's NV12 data."""
  w, h, stride, uv_offset = buf.width, buf.height, buf.stride, buf.uv_offset
  if stride == w and uv_offset == w * h:
    return buf.data[:w * h * 3 // 2]
  a = np.frombuffer(buf.data, dtype=np.uint8)
  y = a[:h * stride].reshape(h, stride)[:, :w]
  uv = a[uv_offset:uv_offset + (h // 2) * stride].reshape(h // 2, stride)[:, :w]
  return np.concatenate([y.reshape(-1), uv.reshape(-1)]).tobytes()
