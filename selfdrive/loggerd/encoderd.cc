#include "selfdrive/loggerd/loggerd.h"

ExitHandler do_exit;

struct EncoderdState {
  int max_waiting = 0;

  // Sync logic for startup
  std::atomic<int> encoders_ready = 0;
  std::atomic<uint32_t> start_frame_id = 0;
  bool camera_ready[WideRoadCam + 1] = {};
  bool camera_synced[WideRoadCam + 1] = {};
};

// Handle initial encoder syncing by waiting for all encoders to reach the same frame id
bool sync_encoders(EncoderdState *s, CameraType cam_type, uint32_t frame_id) {
  if (s->camera_synced[cam_type]) return true;

  if (s->max_waiting > 1 && s->encoders_ready != s->max_waiting) {
    // add a small margin to the start frame id in case one of the encoders already dropped the next frame
    update_max_atomic(s->start_frame_id, frame_id + 2);
    if (std::exchange(s->camera_ready[cam_type], true) == false) {
      ++s->encoders_ready;
      LOGD("camera %d encoder ready", cam_type);
    }
    return false;
  } else {
    if (s->max_waiting == 1) update_max_atomic(s->start_frame_id, frame_id);
    bool synced = frame_id >= s->start_frame_id;
    s->camera_synced[cam_type] = synced;
    if (!synced) LOGD("camera %d waiting for frame %d, cur %d", cam_type, (int)s->start_frame_id, frame_id);
    return synced;
  }
}


void encoder_thread(EncoderdState *s, const LogCameraInfo &cam_info) {
  util::set_thread_name(cam_info.filename);

  std::vector<Encoder *> encoders;
  VisionIpcClient vipc_client = VisionIpcClient("camerad", cam_info.stream_type, false);

  int cur_seg = 0;
  while (!do_exit) {
    if (!vipc_client.connect(false)) {
      util::sleep_for(5);
      continue;
    }

    // init encoders
    if (encoders.empty()) {
      VisionBuf buf_info = vipc_client.buffers[0];
      LOGW("encoder %s init %dx%d", cam_info.filename, buf_info.width, buf_info.height);

      if (buf_info.width > 0 && buf_info.height > 0) {
        // main encoder
        encoders.push_back(new Encoder(cam_info.filename, cam_info.type, buf_info.width, buf_info.height,
                                      cam_info.fps, cam_info.bitrate,
                                      cam_info.is_h265 ? cereal::EncodeIndex::Type::FULL_H_E_V_C : cereal::EncodeIndex::Type::QCAMERA_H264,
                                      buf_info.width, buf_info.height, false));
        // qcamera encoder
        if (cam_info.has_qcamera) {
          encoders.push_back(new Encoder(qcam_info.filename, cam_info.type, buf_info.width, buf_info.height,
                                        qcam_info.fps, qcam_info.bitrate,
                                        qcam_info.is_h265 ? cereal::EncodeIndex::Type::FULL_H_E_V_C : cereal::EncodeIndex::Type::QCAMERA_H264,
                                        qcam_info.frame_width, qcam_info.frame_height, false));
        }
      } else {
        LOGE("not initting empty encoder");
        s->max_waiting--;
        break;
      }
    }

    for (int i = 0; i < encoders.size(); ++i) {
      encoders[i]->encoder_open(NULL);
    }

    bool lagging = false;
    while (!do_exit) {
      VisionIpcBufExtra extra;
      VisionBuf* buf = vipc_client.recv(&extra);
      if (buf == nullptr) continue;

      // detect loop around and drop the frames
      if (buf->get_frame_id() != extra.frame_id) {
        if (!lagging) {
          LOGE("encoder %s lag  buffer id: %d  extra id: %d", cam_info.filename, buf->get_frame_id(), extra.frame_id);
          lagging = true;
        }
        continue;
      }
      lagging = false;

      if (!sync_encoders(s, cam_info.type, extra.frame_id)) {
        continue;
      }
      if (do_exit) break;

      // do rotation if required
      const int frames_per_seg = SEGMENT_LENGTH * MAIN_FPS;
      if (cur_seg >= 0 && extra.frame_id >= ((cur_seg + 1) * frames_per_seg) + s->start_frame_id) {
        for (auto &e : encoders) {
          e->encoder_close();
          e->encoder_open(NULL);
        }
        ++cur_seg;
      }

      // encode a frame
      for (int i = 0; i < encoders.size(); ++i) {
        int out_id = encoders[i]->encode_frame(buf, &extra);

        if (out_id == -1) {
          LOGE("Failed to encode frame. frame_id: %d", extra.frame_id);
        }
      }
    }
  }

  LOG("encoder destroy");
  for(auto &e : encoders) {
    e->encoder_close();
    delete e;
  }
}

void encoderd_thread() {
  EncoderdState s;

  std::vector<std::thread> encoder_threads;
  for (const auto &cam : cameras_logged) {
    encoder_threads.push_back(std::thread(encoder_thread, &s, cam));
    s.max_waiting++;
  }
  for (auto &t : encoder_threads) t.join();
}

int main() {
  if (!Hardware::PC()) {
    int ret;
    ret = util::set_realtime_priority(52);
    assert(ret == 0);
    ret = util::set_core_affinity({3});
    assert(ret == 0);
  }
  encoderd_thread();
  return 0;
}
