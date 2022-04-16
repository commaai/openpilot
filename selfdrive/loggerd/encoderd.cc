#include <atomic>
#include <vector>

#include "selfdrive/common/swaglog.h"

#if defined(QCOM) || defined(QCOM2)
#include "selfdrive/loggerd/omx_encoder.h"
#define Encoder OmxEncoder
#else
#include "selfdrive/loggerd/raw_logger.h"
#define Encoder RawLogger
#endif

ExitHandler do_exit;

struct EncoderdState {
  std::mutex rotate_lock;
  std::condition_variable rotate_cv;
  std::atomic<int> rotate_segment;
  std::atomic<double> last_camera_seen_tms;
  std::atomic<int> ready_to_rotate;  // count of encoders ready to rotate
  int max_waiting = 0;

  // Sync logic for startup
  std::atomic<int> encoders_ready = 0;
  std::atomic<uint32_t> start_frame_id = 0;
  bool camera_ready[WideRoadCam + 1] = {};
  bool camera_synced[WideRoadCam + 1] = {};
};

bool trigger_rotate_if_needed(EncoderdState *s, int cur_seg, uint32_t frame_id) {
  const int frames_per_seg = SEGMENT_LENGTH * MAIN_FPS;
  if (cur_seg >= 0 && frame_id >= ((cur_seg + 1) * frames_per_seg) + s->start_frame_id) {
    // trigger rotate and wait until the main logger has rotated to the new segment
    ++s->ready_to_rotate;
    std::unique_lock lk(s->rotate_lock);
    s->rotate_cv.wait(lk, [&] {
      return s->rotate_segment > cur_seg || do_exit;
    });
    return !do_exit;
  }
  return false;
}

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

  int cur_seg = -1;
  int encode_idx = 0;
  std::vector<Encoder *> encoders;
  VisionIpcClient vipc_client = VisionIpcClient("camerad", cam_info.stream_type, false);

  // While we write them right to the log for sync, we also publish the encode idx to the socket
  const char *service_name = cam_info.type == DriverCam ? "driverEncodeIdx" : (cam_info.type == WideRoadCam ? "wideRoadEncodeIdx" : "roadEncodeIdx");
  PubMaster pm({service_name});

  while (!do_exit) {
    if (!vipc_client.connect(false)) {
      util::sleep_for(5);
      continue;
    }

    // init encoders
    if (encoders.empty()) {
      VisionBuf buf_info = vipc_client.buffers[0];
      LOGD("encoder init %dx%d", buf_info.width, buf_info.height);

      // main encoder
      encoders.push_back(new Encoder(cam_info.filename, cam_info.type, buf_info.width, buf_info.height,
                                     cam_info.fps, cam_info.bitrate, cam_info.is_h265,
                                     cam_info.downscale, false));
      // qcamera encoder
      if (cam_info.has_qcamera) {
        encoders.push_back(new Encoder(qcam_info.filename, cam_info.type, qcam_info.frame_width, qcam_info.frame_height,
                                       qcam_info.fps, qcam_info.bitrate, qcam_info.is_h265, qcam_info.downscale, false));
      }
    }

    while (!do_exit) {
      VisionIpcBufExtra extra;
      VisionBuf* buf = vipc_client.recv(&extra);
      if (buf == nullptr) continue;

      if (cam_info.trigger_rotate) {
        s->last_camera_seen_tms = millis_since_boot();
        if (!sync_encoders(s, cam_info.type, extra.frame_id)) {
          continue;
        }

        // check if we're ready to rotate
        trigger_rotate_if_needed(s, cur_seg, extra.frame_id);
        if (do_exit) break;
      }

      // rotate the encoder if the logger is on a newer segment
      if (s->rotate_segment > cur_seg) {
        cur_seg = s->rotate_segment;

        LOGW("camera %d rotate encoder", cam_info.type);
        for (auto &e : encoders) {
          e->encoder_close();
          e->encoder_open(NULL);
        }
      }

      // encode a frame
      for (int i = 0; i < encoders.size(); ++i) {
        int out_id = encoders[i]->encode_frame(buf->y, buf->u, buf->v,
                                               buf->width, buf->height, extra.timestamp_eof);

        if (out_id == -1) {
          LOGE("Failed to encode frame. frame_id: %d encode_id: %d", extra.frame_id, encode_idx);
        }

        // publish encode index
        if (i == 0 && out_id != -1) {
          MessageBuilder msg;
          // this is really ugly
          bool valid = (buf->get_frame_id() == extra.frame_id);
          auto eidx = cam_info.type == DriverCam ? msg.initEvent(valid).initDriverEncodeIdx() :
                     (cam_info.type == WideRoadCam ? msg.initEvent(valid).initWideRoadEncodeIdx() : msg.initEvent(valid).initRoadEncodeIdx());
          eidx.setFrameId(extra.frame_id);
          eidx.setTimestampSof(extra.timestamp_sof);
          eidx.setTimestampEof(extra.timestamp_eof);
          if (Hardware::TICI()) {
            eidx.setType(cereal::EncodeIndex::Type::FULL_H_E_V_C);
          } else {
            eidx.setType(cam_info.type == DriverCam ? cereal::EncodeIndex::Type::FRONT : cereal::EncodeIndex::Type::FULL_H_E_V_C);
          }
          eidx.setEncodeId(encode_idx);
          eidx.setSegmentNum(cur_seg);
          eidx.setSegmentId(out_id);
          pm.send(service_name, msg);
        }
      }

      encode_idx++;
    }
  }

  LOG("encoder destroy");
  for(auto &e : encoders) {
    e->encoder_close();
    delete e;
  }
}

int main(int argc, char** argv) {
  EncoderdState s;

  // init encoders
  s.last_camera_seen_tms = millis_since_boot();
  std::vector<std::thread> encoder_threads;
  for (const auto &cam : cameras_logged) {
    if (cam.enable) {
      encoder_threads.push_back(std::thread(encoder_thread, &s, cam));
      if (cam.trigger_rotate) s.max_waiting++;
    }
  }

  LOGW("closing encoders");
  s.rotate_cv.notify_all();
  for (auto &t : encoder_threads) t.join();

  return 0;
}