#include <future>

#include "catch2/catch.hpp"
#include "selfdrive/common/util.h"
#include "selfdrive/loggerd/logger.h"
#include "selfdrive/loggerd/logging.h"

TEST_CASE("sync encoders") {
  auto thread_func = [](LoggerdState *s, CameraType cam_type, int frame_id) -> int {
    while (!sync_encoders(s, cam_type, frame_id)) {
      ++frame_id;
      util::sleep_for(0);
    }
    return frame_id;
  };

  LoggerdState s;
  s.max_waiting = MAX_CAMERAS;
  int start_frame_id[MAX_CAMERAS] = {10, 20, 30};
  std::future<int> futures[MAX_CAMERAS];
  for (int i = 0; i < MAX_CAMERAS; ++i) {
    futures[i] = std::async(std::launch::async, thread_func, &s, (CameraType)i, start_frame_id[i]);
  }
  int start_id = *std::max_element(start_frame_id, start_frame_id + MAX_CAMERAS) + 2;
  for (auto &f : futures) {
    REQUIRE(f.get() == start_id);
  }
}
