#include <future>

#include "catch2/catch.hpp"
#include "selfdrive/common/util.h"
#include "selfdrive/loggerd/logger.h"
#include "selfdrive/loggerd/logging.h"

TEST_CASE("sync encoders") {
  auto thread_func = [](LoggerdState *s, CameraType cam_type) -> int {
    srand( time(nullptr));
    int frame_id = rand() % 10;
    while (!sync_encoders(s, cam_type, frame_id)) {
      ++frame_id;
      usleep(0);
    }
    return frame_id;
  };

  LoggerdState s{.max_waiting = MAX_CAMERAS};
  std::future<int> futures[MAX_CAMERAS];
  for (int i = 0; i < MAX_CAMERAS; ++i) {
    futures[i] = std::async(std::launch::async, thread_func, &s, (CameraType)i);
  }
  int start_frames[] = {futures[0].get(), futures[1].get(), futures[2].get()};
  REQUIRE(start_frames[0] == start_frames[1]);
  REQUIRE(start_frames[1] == start_frames[2]);
  
}
