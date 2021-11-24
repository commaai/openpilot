#include <future>

#include "catch2/catch.hpp"
#include "selfdrive/loggerd/loggerd.h"

int random_int(int min, int max) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(min, max);
  return dist(rng);
}

int get_synced_frame_id(LoggerdState *s, CameraType cam_type, int start_frame_id) {
  int frame_id = start_frame_id;
  while (!sync_encoders(s, cam_type, frame_id)) {
    ++frame_id;
    usleep(0);
  }
  return frame_id;
};

TEST_CASE("sync_encoders") {
  const int max_waiting = GENERATE(1, 2, 3);

  for (int test_cnt = 0; test_cnt < 10; ++test_cnt) {
    LoggerdState s{.max_waiting = max_waiting};
    std::vector<int> start_frames;
    std::vector<std::future<int>> futures;

    for (int i = 0; i < max_waiting; ++i) {
      int start_frame_id = random_int(0, 20);
      start_frames.push_back(start_frame_id);
      futures.emplace_back(std::async(std::launch::async, get_synced_frame_id, &s, (CameraType)i, start_frame_id));
    }

    // get results
    int synced_frame_id = 0;
    for (int i = 0; i < max_waiting; ++i) {
      if (i == 0) {
        synced_frame_id = futures[i].get();
        // require synced_frame_id equal start_frame_id if max_waiting == 1
        if (max_waiting == 1) {
          REQUIRE(synced_frame_id == start_frames[0]);
        }
      } else {
        REQUIRE(futures[i].get() == synced_frame_id);
      }
    }
  }
}
