#include <future>

#include "catch2/catch.hpp"
#include "selfdrive/common/util.h"
#include "selfdrive/loggerd/logger.h"
#include "selfdrive/loggerd/loggerd.h"

const int MAX_CAMERAS = 3;

TEST_CASE("sync_encoders") {
  auto thread_func = [](LoggerdState *s, CameraType cam_type) -> int {
    srand(time(nullptr));
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

  // get results
  std::vector<int> synced_frame_ids;
  for (int i = 0; i < MAX_CAMERAS; ++i) {
    synced_frame_ids.push_back(futures[i].get());
  }
  REQUIRE(std::equal(synced_frame_ids.begin() + 1, synced_frame_ids.end(), synced_frame_ids.begin()));
}
