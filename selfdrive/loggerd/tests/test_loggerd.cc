#include "catch2/catch.hpp"
#include "selfdrive/loggerd/logging.h"

ExitHandler do_exit;
const int ROTATE_CNT = 1000;

void encode_thread(LoggerdState *s, bool trigger_rotate, bool test_timeout_rotate) {
  int cnt = 0, rotated_cnt = 0, cur_seg = -1;
  do {
    if (trigger_rotate) {
      if (cnt >= 50) {
        REQUIRE(s->rotate_segment == cur_seg);
        s->triggerAndWait(cur_seg, &do_exit);
        if (!do_exit) {
          REQUIRE(s->rotate_segment == cur_seg + 1);
          REQUIRE(s->waiting_rotate == 0);
        }
      } else {
        REQUIRE(s->rotate_segment == cur_seg);
      }
    }
    if (s->rotate_segment > cur_seg) {
      REQUIRE(s->rotate_segment == cur_seg + 1);
      ++rotated_cnt;
      cnt = 0;
      cur_seg = s->rotate_segment;
    }
    ++cnt;
    usleep(0);
  } while (!do_exit);
  REQUIRE(rotated_cnt == ROTATE_CNT);
  REQUIRE(cur_seg == s->rotate_segment);
}

TEST_CASE("log rotation") {
  LoggerdState s;
  logger_init(&s.logger, "rlog", true);
  s.rotate(true);
  s.max_waiting = 2;
  std::vector<std::thread> threads;
  for (int i = 0; i < 3; ++i) {
    threads.push_back(std::thread(encode_thread, &s, i != 0));
  }

  while (!do_exit) {
    s.rotate_if_needed(true);
    if (s.rotate_segment == ROTATE_CNT - 1) break;

    util::sleep_for(1);
  }
  do_exit = true;
  s.rotate_cv.notify_all();
  for (auto &t : threads) t.join();

  logger_close(&s.logger);
}
