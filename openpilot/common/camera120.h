#pragma once

#include <cstdlib>
#include <cstring>

constexpr int CAMERA_FPS = 60;

// Legacy name retained for existing deployments; both flags select 60 FPS.
inline bool camera120_enabled() {
  const char *value = std::getenv("CAMERA_720P60");
  if (!value) value = std::getenv("CAMERA_720P120");
  return value && std::strcmp(value, "1") == 0;
}
