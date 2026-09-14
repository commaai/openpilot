#pragma once

#include <cstdlib>
#include <cstring>

// Selected by the bench runner or the manager's opt-in startup configuration.
inline bool camera120_enabled() {
  const char *value = std::getenv("CAMERA_720P120");
  return value && std::strcmp(value, "1") == 0;
}
