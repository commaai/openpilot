#include "selfdrive/loggerd/logging.h"

LoggerdState::LoggerdState(int segment_length_ms, int no_camera_patience, bool testing)
    : segment_length_ms(segment_length_ms), no_camera_patience(no_camera_patience), testing(testing) {
}

std::optional<std::pair<int, std::string>> LoggerdState::get_segment(int cur_seg, bool trigger_rotate, ExitHandler *do_exit) {
  std::unique_lock lk(rotate_lock);
  if (trigger_rotate) {
    ++waiting_rotate;
    rotate_cv.wait(lk, [&] { return rotate_segment > cur_seg || *do_exit; });
  }
  return (*do_exit) ? std::nullopt : std::make_optional(std::pair{rotate_segment, segment_path});
}

void LoggerdState::rotate() {
  if (!testing) {
    int err = logger_next(&logger, LOG_ROOT.c_str(), segment_path, sizeof(segment_path), &rotate_segment);
    assert(err == 0);
  } else {
    rotate_segment += 1;
    snprintf(segment_path, sizeof(segment_path), "%s/%s--%d", LOG_ROOT.c_str(), logger.route_name.c_str(), rotate_segment);
  }
  last_rotate_tms = last_camera_seen_tms = millis_since_boot();
  LOGW((logger.part == 0) ? "logging to %s" : "rotated to %s", segment_path);
}

bool LoggerdState::rotate_if_needed() {
  std::unique_lock lk(rotate_lock);
  bool do_rotate = waiting_rotate == max_waiting;
  if (!do_rotate) {
    double tms = millis_since_boot();
    if ((tms - last_rotate_tms) > segment_length_ms &&
        (tms - last_camera_seen_tms) > no_camera_patience &&
        !LOGGERD_TEST) {
      LOGW("no camera packet seen. auto rotating");
      do_rotate = true;
    }
  }
  if (do_rotate) {
    rotate();
    waiting_rotate = 0;
    rotate_cv.notify_all();
  }
  return do_rotate;
}
