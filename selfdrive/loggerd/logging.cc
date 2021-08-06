#include "selfdrive/loggerd/logging.h"

LoggerdState::LoggerdState(int segment_length_ms, int no_camera_patience, bool testing)
    : segment_length_ms(segment_length_ms), no_camera_patience(no_camera_patience), testing(testing) {
}

void LoggerdState::rotate() {
  {
    std::unique_lock lk(rotate_lock);
    int segment = -1;
    if (!testing) {
      int err = logger_next(&logger, LOG_ROOT.c_str(), segment_path, sizeof(segment_path), &segment);
      assert(err == 0);
    } else {
      segment = rotate_segment + 1;
      snprintf(segment_path, sizeof(segment_path), "%s/%s--%d", LOG_ROOT.c_str(), logger.route_name.c_str(), segment);
    }
    rotate_segment = segment;
    waiting_rotate = 0;
    last_rotate_tms = millis_since_boot();
  }
  rotate_cv.notify_all();
  LOGW((rotate_segment == 0) ? "logging to %s" : "rotated to %s", segment_path);
}

void LoggerdState::rotate_if_needed() {
  if (waiting_rotate == max_waiting) {
    rotate();
  }

  double tms = millis_since_boot();
  if ((tms - last_rotate_tms) > segment_length_ms &&
      (tms - last_camera_seen_tms) > no_camera_patience) {
    LOGW("no camera packet seen. auto rotating");
    rotate();
  }
}

void LoggerdState::triggerAndWait(int cur_seg, ExitHandler *do_exit) {
  // trigger rotate and wait logger rotated to new segment
  ++waiting_rotate;
  std::unique_lock lk(rotate_lock);
  rotate_cv.wait(lk, [&] { return rotate_segment > cur_seg || *do_exit; });
}
