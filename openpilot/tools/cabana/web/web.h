#pragma once

#include <string>

#include "tools/replay/util.h"

namespace web {
void setApiToken(const std::string &token);
bool hasApiToken();

namespace video {
// the newest decoded frame of a camera, as a GL texture owned by the page. Main thread only.
struct Frame {
  unsigned int texture = 0;
  int width = 0;
  int height = 0;
};
Frame latest(CameraType type);
}  // namespace video
}  // namespace web
