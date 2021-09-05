#include "catch2/catch.hpp"
#include "selfdrive/loggerd/logging.h"

ExitHandler do_exit;

const int segment_length = 10;
const int no_camera_patience = 6;  // ms

const int ENCODER_THREAD_CNT = 10;
const int ROTATE_CNT = 200;

// catch2's s macros are not thread safe.
std::mutex catch2_lock;
#define SAFE_REQUIRE(x)              \
  {                                  \
    std::lock_guard lk(catch2_lock); \
    REQUIRE(x);                      \
  }

#define SAFE_UNSCOPED_INFO(x)        \
  {                                  \
    std::lock_guard lk(catch2_lock); \
    UNSCOPED_INFO(x);                \
  }

// helper class to accessing protected varialbes of LoggerdState
class TestLoggerdState : public LoggerdState {
 public:
  TestLoggerdState(int segment_length_ms, int no_camera_patience, bool testing)
      : LoggerdState(segment_length_ms, no_camera_patience, testing) {}

  friend void test_rotation(bool has_camera);
  friend void encode_thread(TestLoggerdState *s, bool trigger_rotate, bool has_camera);
};

void encode_thread(TestLoggerdState *s, bool trigger_rotate, bool has_camera) {
  int cnt = 0, rotated_cnt = 0, cur_seg = -1;
  while (!do_exit) {
    if (trigger_rotate && has_camera) {
      s->last_camera_seen_tms = millis_since_boot();
    }
    auto segment = s->get_segment(cur_seg, trigger_rotate && cnt >= segment_length, &do_exit);
    if (!segment) break;

    auto [segment_id, segment_path] = *segment;
    // rotate if the logger is on a newer segment
    if (segment_id > cur_seg) {
      SAFE_REQUIRE(s->waiting_rotate == 0);
      SAFE_REQUIRE(segment_id == cur_seg + 1);
      if (trigger_rotate && cur_seg != -1 && has_camera) {
        SAFE_REQUIRE(cnt == segment_length);
      }
      ++rotated_cnt;
      cnt = 0;
      cur_seg = segment_id;
    }
    cnt += has_camera;
    util::sleep_for(1);
  };

  SAFE_UNSCOPED_INFO("thread [trigger_rotate=" << trigger_rotate << "][has_camera=" << has_camera << "]");
  SAFE_REQUIRE(rotated_cnt == ROTATE_CNT);
  SAFE_REQUIRE(cur_seg == s->rotate_segment);
}

void test_rotation(bool has_camera) {
  TestLoggerdState s(segment_length, no_camera_patience, true);
  s.rotate();
  std::vector<std::thread> threads;
  for (int i = 0; i < ENCODER_THREAD_CNT; ++i) {
    bool trigger_rotate = i > 0;
    threads.push_back(std::thread(encode_thread, &s, trigger_rotate, has_camera));
    s.max_waiting += trigger_rotate;
  }

  for (int rotated = 1; rotated < ROTATE_CNT; /**/) {
    double last_camera_seen_tms = s.last_camera_seen_tms;
    if (s.rotate_if_needed()) {
      SAFE_REQUIRE(s.waiting_rotate == 0);
      if (!has_camera) {
        // make sure this is a timeout rotation
        SAFE_REQUIRE((millis_since_boot() - last_camera_seen_tms) > no_camera_patience);
      }
      rotated++;
    } else {
      SAFE_REQUIRE(s.waiting_rotate <= s.max_waiting);
      if (!has_camera) {
        SAFE_REQUIRE(uint64_t(millis_since_boot() - s.last_rotate_tms) <= segment_length);
      }
    }
    util::sleep_for(1);
  }

  // wait threads finished
  do_exit = true;
  util::sleep_for(20);
  s.rotate_cv.notify_all();
  for (auto &t : threads) t.join();
}

TEST_CASE("log rotation") {
  do_exit = false;
  SECTION("test rotation with camera") {
    test_rotation(true);
  }
  SECTION("test rotation without camera") {
    test_rotation(false);
  }
}

TEST_CASE("sync encoders") {
  auto thread_func = [](LoggerdState *s, CameraType cam_type, int frame_id, int start_frame[]) {
    while (!s->sync_encoders(cam_type, frame_id)) {
      ++frame_id;
      util::sleep_for(1);
    }
    start_frame[cam_type] = frame_id;
  };

  LoggerdState s;
  s.max_waiting = 3;
  int encoder_start_frame[MAX_CAMERAS] = {};
  std::vector<std::thread> threads;
  for (int i = 0; i < MAX_CAMERAS; ++i) {
    threads.emplace_back(thread_func, &s, (CameraType)i, i, encoder_start_frame);
  }
  for (auto &t : threads) t.join();

  REQUIRE(encoder_start_frame[RoadCam] == encoder_start_frame[DriverCam]);
  REQUIRE(encoder_start_frame[RoadCam] == encoder_start_frame[WideRoadCam]);
}
