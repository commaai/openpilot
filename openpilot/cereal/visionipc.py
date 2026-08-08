"""openpilot's camera streams, keyed by visionipc stream id.

msgq's visionipc treats stream types as opaque ids; the mapping from
stream id to camera lives here so msgq stays camera-agnostic.
"""
from enum import IntEnum


class VisionStreamType(IntEnum):
  VISION_STREAM_ROAD = 0
  VISION_STREAM_DRIVER = 1
  VISION_STREAM_WIDE_ROAD = 2
  VISION_STREAM_MAP = 3
