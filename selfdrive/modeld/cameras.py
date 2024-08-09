import time
from msgq.visionipc import VisionIpcClient, VisionStreamType
from openpilot.common.swaglog import cloudlog


class Cameras:
  def __init__(self, context):
    self.main_stream_type, self.extra_stream_type = self._get_stream_types()
    self.main = self._init_client(self.main_stream_type, True, context)
    self.extra = self._init_client(self.extra_stream_type, False, context) if self.extra_stream_type else self.main

  def recv(self):
    if not (buf_main := self.main.recv()):
      return None, None

    if self.extra == self.main:
      return buf_main, buf_main

    while (buf_extra := self.extra.recv()):
      if self.extra.frame_id == self.main.frame_id:
        return buf_main, buf_extra

    return None, None

  def _get_stream_types(self):
    while not (available_streams := VisionIpcClient.available_streams("camerad", block=False)):
      time.sleep(0.1)

    has_road = VisionStreamType.VISION_STREAM_ROAD in available_streams
    has_wide = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams
    main_stream_type = VisionStreamType.VISION_STREAM_ROAD if has_road else VisionStreamType.VISION_STREAM_WIDE_ROAD
    extra_stream_type = VisionStreamType.VISION_STREAM_WIDE_ROAD if has_road and has_wide else None
    return main_stream_type, extra_stream_type

  def _init_client(self, stream_type, conflate, context):
    client = VisionIpcClient("camerad", stream_type, conflate, context)
    while not client.connect(False):
      time.sleep(0.1)

    cloudlog.warning(f"connected cam with buffer size: {client.buffer_len} ({client.width} x {client.height})")
    return client
