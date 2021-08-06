#include "selfdrive/loggerd/logging.h"

void LoggerdState::rotate(bool fake_rotate) {
  {
    std::unique_lock lk(rotate_lock);
    int segment = -1;
    if (!fake_rotate) {
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

void LoggerdState::rotate_if_needed(bool fake_rotate) {
  if (waiting_rotate == max_waiting) {
    rotate(fake_rotate);
  }

  double tms = millis_since_boot();
  if ((tms - last_rotate_tms) > SEGMENT_LENGTH * 1000 &&
      (tms - last_camera_seen_tms) > NO_CAMERA_PATIENCE) {
    LOGW("no camera packet seen. auto rotating");
    rotate(fake_rotate);
  }
}

void LoggerdState::triggerAndWait(int cur_seg, ExitHandler *do_exit) {
  // trigger rotate and wait logger rotated to new segment
  ++waiting_rotate;
  std::unique_lock lk(rotate_lock);
  rotate_cv.wait(lk, [&] { return rotate_segment > cur_seg || *do_exit; });
}
