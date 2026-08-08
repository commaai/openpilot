#pragma once

#include "msgq/visionipc/visionbuf.h"

// openpilot's camera streams, keyed by visionipc stream id.
// msgq's visionipc treats stream types as opaque ids; the mapping from
// stream id to camera lives here so msgq stays camera-agnostic.
// Keep in sync with cereal/visionipc.py
enum VisionStreamValues : VisionStreamType {
  VISION_STREAM_ROAD = 0,
  VISION_STREAM_DRIVER = 1,
  VISION_STREAM_WIDE_ROAD = 2,
  VISION_STREAM_MAP = 3,
};
