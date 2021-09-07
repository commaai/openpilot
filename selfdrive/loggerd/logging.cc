#include "selfdrive/loggerd/logging.h"
// Wait for all encoders to reach the same frame id
bool sync_encoders(LoggerdState *s, CameraType cam_type, uint32_t frame_id) {
  if (s->max_waiting > 1 && s->encoders_ready != s->max_waiting) {
    if (std::exchange(s->camera_ready[cam_type], true) == false) {
      ++s->encoders_ready;
      LOGE("camera %d encoder ready", cam_type);
    }
    if (s->latest_frame_id < frame_id) {
      s->latest_frame_id = frame_id;
    }
    return false;
  } else {
    // Small margin in case one of the encoders already dropped the next frame
    uint32_t start_frame_id = s->latest_frame_id + 2;
    bool synced = frame_id >= start_frame_id;
    if (!synced) LOGE("camera %d waiting for frame %d, cur %d", cam_type, start_frame_id, frame_id);
    return synced;
  }
}
