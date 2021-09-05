#include "selfdrive/loggerd/logging.h"

LoggerdState::LoggerdState(int segment_length_ms, int no_camera_patience, bool testing)
    : segment_length_ms(segment_length_ms), no_camera_patience(no_camera_patience), testing(testing) {
}

void LoggerdState::init() {
  logger_init(&logger, "rlog", true);
  rotate();
}

void LoggerdState::close(ExitHandler *do_exit) {
  rotate_cv.notify_all();
  LOGW("closing logger");
  logger_close(&logger, do_exit);
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

bool LoggerdState::sync_encoders(CameraType cam_type, uint32_t frame_id) {
  std::unique_lock lk(sync_lock);
  if (max_waiting > 1 && encoders_ready != max_waiting) {
     if (std::exchange(camera_ready[cam_type], true) == false) {
      ++encoders_ready;
      LOGE("camera %d encoder ready", cam_type);
    }
    if (latest_frame_id < frame_id) {
      latest_frame_id = frame_id;
    }
    return false;
  } else {
    // Small margin in case one of the encoders already dropped the next frame
    uint32_t start_frame_id = latest_frame_id + 2;
    bool synced = frame_id >= start_frame_id;
    if (!synced) LOGE("camera %d waiting for frame %d, cur %d", cam_type, start_frame_id, frame_id);
    return synced;
  }
}
