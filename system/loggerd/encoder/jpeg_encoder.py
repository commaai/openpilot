import numpy as np
import jpeglib

import cereal.messaging as messaging


class JpegEncoder:
  def __init__(self, publish_name: str, width: int, height: int):
    self.publish_name = publish_name
    self.width = width
    self.height = height
    self.pm = messaging.PubMaster([publish_name])

  def pushThumbnail(self, buf, extra) -> None:
    W, H, S = buf.width, buf.height, buf.stride
    y_full = np.asarray(buf.data[:buf.uv_offset], dtype=np.uint8).reshape((-1, S))[:H, :W]
    uv_rows = H // 2
    uv_full = np.asarray(buf.data[buf.uv_offset:buf.uv_offset + uv_rows * S], dtype=np.uint8).reshape((uv_rows, S))[:, :W]

    tw, th = self.width, self.height
    if not (W % 2 == 0 and H % 2 == 0 and tw * 2 == W and th * 2 == H):
      return

    y_small = y_full.reshape(th, 2, tw, 2).mean(axis=(1, 3)).astype(np.uint8)
    u_full = uv_full[:, 0::2]
    v_full = uv_full[:, 1::2]
    u_small = u_full.reshape(th // 2, 2, tw // 2, 2).mean(axis=(1, 3)).astype(np.uint8)
    v_small = v_full.reshape(th // 2, 2, tw // 2, 2).mean(axis=(1, 3)).astype(np.uint8)

    jpeg_bytes = jpeglib.encode_YUV420(y_small, u_small, v_small, sampling="420", quality=80)

    msg = messaging.new_message("thumbnail")
    msg.thumbnail.frameId = int(extra.frame_id)
    msg.thumbnail.timestampEof = int(extra.timestamp_eof)
    msg.thumbnail.thumbnail = jpeg_bytes
    self.pm.send(self.publish_name, msg)
