#include <future>

#include "catch2/catch.hpp"
#include "selfdrive/common/util.h"
#include "selfdrive/loggerd/logger.h"
#include "selfdrive/loggerd/loggerd.h"

const int MAX_CAMERAS = 3;

TEST_CASE("sync_encoders") {
  auto thread_func = [](LoggerdState *s, CameraType cam_type) -> int {
    srand(time(nullptr));
    int frame_id = rand() % 20;
    while (!sync_encoders(s, cam_type, frame_id)) {
      ++frame_id;
      usleep(0);
    }
    return frame_id;
  };

  for (int test_cnt = 0; test_cnt < 10; ++test_cnt) {
    LoggerdState s{.max_waiting = MAX_CAMERAS};
    std::future<int> futures[MAX_CAMERAS];
    for (int i = 0; i < MAX_CAMERAS; ++i) {
      futures[i] = std::async(std::launch::async, thread_func, &s, (CameraType)i);
    }

    // get results
    int frame_id = 0;
    for (int i = 0; i < MAX_CAMERAS; ++i) {
      if (i == 0) {
        frame_id = futures[i].get();
      } else {
        REQUIRE(futures[i].get() == frame_id);
      }
    }
  }
}
