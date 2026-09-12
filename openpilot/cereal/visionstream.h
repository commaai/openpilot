#pragma once

#include "msgq/visionipc/visionbuf.h"

enum VisionStreamValues : VisionStreamType {
  VISION_STREAM_NARROW_ROAD = 0,
  VISION_STREAM_CABIN = 1,
  VISION_STREAM_WIDE_ROAD = 2,
  VISION_STREAM_MAP = 3,
};
