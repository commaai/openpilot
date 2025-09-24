import time
import numpy as np
import av as _av
from typing import Any, cast
av: Any = _av


import cereal.messaging as messaging

from openpilot.system.loggerd.encoder.encoder import VideoEncoder


class FfmpegEncoder(VideoEncoder):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.codec: Any | None = None

  def open(self) -> None:
    settings = self.encoder_info.get_settings(self.in_width)
    codec_name = "h264" if settings.encode_type in ("QCAMERA_H264", "LIVESTREAM_H264") else "hevc"
    self.codec = av.CodecContext.create(codec_name, "w")
    self.codec.width = self.out_width
    self.codec.height = self.out_height
    self.codec.time_base = av.time_base.TimeBase(1, self.encoder_info.fps)
    self.codec.framerate = av.time_base.TimeBase(self.encoder_info.fps, 1)
    self.codec.bit_rate = settings.bitrate
    self.codec.options = {"g": str(settings.gop_size), "bf": "0"}
    self.codec.open()
    self.segment_num += 1
    self.counter = 0

  def close(self) -> None:
    if self.codec:
      try:
        for _ in self.codec.encode(None):
          pass
      except Exception:
        pass
      self.codec = None

  def _publish_packet(self, pkt, frame_id: int, timestamp_sof: int, timestamp_eof: int) -> None:
    msg = messaging.new_message(self.encoder_info.publish_name)
    ed = getattr(msg, self.encoder_info.publish_name)
    ed.unixTimestampNanos = int(time.time_ns())
    idx = ed.idx
    idx.frameId = int(frame_id)
    idx.timestampSof = int(timestamp_sof)
    idx.timestampEof = int(timestamp_eof)
    idx.encodeId = self.encode_cnt
    idx.segmentNum = int(self.segment_num)
    idx.segmentId = int(self.counter)
    flags = 8 if getattr(pkt, "is_keyframe", False) else 0
    idx.flags = flags
    data = bytes(pkt)
    idx.len = len(data)
    ed.data = data
    ed.width = self.out_width
    ed.height = self.out_height
    if flags & 8:
      try:
        if self.codec and self.codec.extradata:
          ed.header = bytes(self.codec.extradata)
      except Exception:
        pass
    self.pm.send(self.encoder_info.publish_name, msg)
    self.encode_cnt += 1
    self.counter += 1

  def encode_frame(self, buf, frame_id: int, timestamp_sof: int, timestamp_eof: int) -> int:
    H, W, S = buf.height, buf.width, buf.stride
    y = np.asarray(buf.data[:buf.uv_offset], dtype=np.uint8).reshape((-1, S))[:H, :W].copy(order="C")
    uv_rows = H // 2
    uv = np.asarray(buf.data[buf.uv_offset:buf.uv_offset + uv_rows * S], dtype=np.uint8).reshape((uv_rows, S))[:, :W].copy(order="C")

    frame = av.VideoFrame(W, H, "nv12")
    frame.planes[0].update(cast(bytes, y))
    frame.planes[1].update(cast(bytes, uv))

    if self.codec and self.codec.pix_fmt.name != "nv12":
      frame = frame.reformat(format=self.codec.pix_fmt.name, width=self.out_width, height=self.out_height)

    ret = 0
    assert self.codec is not None
    for pkt in self.codec.encode(frame):
      self._publish_packet(pkt, frame_id, timestamp_sof, timestamp_eof)
      ret = self.counter
    return ret
