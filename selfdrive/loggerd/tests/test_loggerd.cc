#include "catch2/catch.hpp"
#include "selfdrive/loggerd/logging.h"

ExitHandler do_exit;

const int thread_cnt = 10;
const int ROTATE_CNT = 100;
const int segment_length_ms = 10;  // ms
const int no_camera_patience = 5;  // ms

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

void encode_thread(LoggerdState *s, bool trigger_rotate, bool has_camera) {
  int cnt = 0, rotated_cnt = 0, cur_seg = -1;
  do {
    if (trigger_rotate) {
      if (has_camera) {
        s->last_camera_seen_tms = millis_since_boot();
      }
      if (cnt >= segment_length_ms) {
        // trigger rotate and wait logger rotated to new segment
        SAFE_REQUIRE(s->rotate_segment == cur_seg);
        s->triggerAndWait(cur_seg, &do_exit);
        if (!do_exit) {
          SAFE_REQUIRE(s->waiting_rotate == 0);
          SAFE_REQUIRE(s->rotate_segment == cur_seg + 1);
        }
      }
    }

    // rotate if the logger is on a newer segment
    if (s->rotate_segment > cur_seg) {
      SAFE_REQUIRE(s->rotate_segment == cur_seg + 1);
      if (trigger_rotate && cur_seg != -1 && has_camera) {
        SAFE_REQUIRE(cnt == segment_length_ms);
      }
      ++rotated_cnt;
      cnt = 0;
      cur_seg = s->rotate_segment;
    }
    cnt += has_camera;
    util::sleep_for(1);
  } while (!do_exit);
  SAFE_UNSCOPED_INFO("thread [trigger_rotate=" << trigger_rotate << "][has_camera=" << has_camera << "]");
  SAFE_REQUIRE(rotated_cnt == ROTATE_CNT);
  SAFE_REQUIRE(cur_seg == s->rotate_segment);
}

void test_rotation(bool has_camera) {
  LoggerdState s(segment_length_ms, no_camera_patience, true);
  logger_init(&s.logger, "rlog", true);
  s.rotate();
  s.max_waiting = thread_cnt - 1;
  s.last_camera_seen_tms = millis_since_boot();
  std::vector<std::thread> threads;
  for (int i = 0; i < thread_cnt; ++i) {
    threads.push_back(std::thread(encode_thread, &s, i != 0, has_camera));
  }

  while (!do_exit) {
    int prev_segment = s.rotate_segment;
    s.rotate_if_needed();

    if (s.rotate_segment != prev_segment) {
      SAFE_REQUIRE(s.waiting_rotate == 0);
      if (!has_camera) {
        // make sure this is a timeout rotation
        SAFE_REQUIRE((millis_since_boot() - s.last_camera_seen_tms) > no_camera_patience);
      }
    } else {
      SAFE_REQUIRE(s.waiting_rotate <= s.max_waiting);
      if (!has_camera) {
        double tms = millis_since_boot();
        SAFE_REQUIRE(uint64_t(tms - s.last_rotate_tms) <= segment_length_ms);
      }
    }

    if (s.rotate_segment == ROTATE_CNT - 1) {
      break;
    }
    util::sleep_for(1);
  }

  // wait threads finished
  util::sleep_for(10);
  do_exit = true;
  s.rotate_cv.notify_all();
  for (auto &t : threads) t.join();

  logger_close(&s.logger);
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
