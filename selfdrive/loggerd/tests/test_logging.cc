#include "catch2/catch.hpp"
#include "selfdrive/common/util.h"
#include "selfdrive/loggerd/logger.h"
#include "selfdrive/loggerd/logging.h"

TEST_CASE("sync encoders") {
  auto thread_func = [](LoggerdState *s, CameraType cam_type, int frame_id, int start_frame[]) {
    while (!sync_encoders(s, cam_type, frame_id)) {
      ++frame_id;
      util::sleep_for(0);
    }
    start_frame[cam_type] = frame_id;
  };

  LoggerdState s;
  s.max_waiting = MAX_CAMERAS;
  int encoder_start_frame[MAX_CAMERAS] = {};
  int start_frame_id[MAX_CAMERAS] = {10, 20, 30};
  std::vector<std::thread> threads;
  for (int i = 0; i < MAX_CAMERAS; ++i) {
    threads.emplace_back(thread_func, &s, (CameraType)i, start_frame_id[i], encoder_start_frame);
  }
  for (auto &t : threads) t.join();

  int start_id = *std::max_element(start_frame_id, start_frame_id + MAX_CAMERAS) + 2;
  REQUIRE(encoder_start_frame[RoadCam] == start_id);
  REQUIRE(encoder_start_frame[DriverCam] == start_id);
  REQUIRE(encoder_start_frame[WideRoadCam] == start_id);
}
