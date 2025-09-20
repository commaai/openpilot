from dataclasses import dataclass

import cereal.messaging as messaging


@dataclass
class EncoderSettings:
  encode_type: str
  bitrate: int
  gop_size: int
  b_frames: int = 0


class VideoEncoder:
  def __init__(self, encoder_info, in_width: int, in_height: int):
    self.encoder_info = encoder_info
    self.in_width = in_width
    self.in_height = in_height
    self.out_width = encoder_info.frame_width if encoder_info.frame_width > 0 else in_width
    self.out_height = encoder_info.frame_height if encoder_info.frame_height > 0 else in_height
    self.pm = messaging.PubMaster([encoder_info.publish_name])
    self.encode_cnt = 0
    self.segment_num = -1
    self.counter = 0

  def open(self) -> None:
    raise NotImplementedError

  def close(self) -> None:
    raise NotImplementedError

  def encode_frame(self, buf, frame_id: int, timestamp_sof: int, timestamp_eof: int) -> int:
    raise NotImplementedError
