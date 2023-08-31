#include <cassert>

#include "system/loggerd/loggerd.h"

#ifdef QCOM2
#include "system/loggerd/encoder/v4l_encoder.h"
#define Encoder V4LEncoder
#else
#include "system/loggerd/encoder/ffmpeg_encoder.h"
#define Encoder FfmpegEncoder
#endif

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
  util::set_thread_name(cam_info.thread_name);

  std::vector<std::unique_ptr<Encoder>> encoders;
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
      LOGW("encoder %s init %zux%zu", cam_info.thread_name, buf_info.width, buf_info.height);
      assert(buf_info.width > 0 && buf_info.height > 0);

      for (const auto &encoder_info : cam_info.encoder_infos) {
        auto &e = encoders.emplace_back(new Encoder(encoder_info, buf_info.width, buf_info.height));
        e->encoder_open(nullptr);
      }
    }

    bool lagging = false;
    while (!do_exit) {
      VisionIpcBufExtra extra;
      VisionBuf* buf = vipc_client.recv(&extra);
      if (buf == nullptr) continue;

      // detect loop around and drop the frames
      if (buf->get_frame_id() != extra.frame_id) {
        if (!lagging) {
          LOGE("encoder %s lag  buffer id: %" PRIu64 " extra id: %d", cam_info.thread_name, buf->get_frame_id(), extra.frame_id);
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
}

template <size_t N>
void encoderd_thread(const LogCameraInfo (&cameras)[N]) {
  EncoderdState s;

  std::set<VisionStreamType> streams;
  while (!do_exit) {
    streams = VisionIpcClient::getAvailableStreams("camerad", false);
    if (!streams.empty()) {
      break;
    }
    util::sleep_for(100);
  }

  if (!streams.empty()) {
    std::vector<std::thread> encoder_threads;
    for (auto stream : streams) {
      auto it = std::find_if(std::begin(cameras), std::end(cameras),
                             [stream](auto &cam) { return cam.stream_type == stream; });
      assert(it != std::end(cameras));
      ++s.max_waiting;
      encoder_threads.push_back(std::thread(encoder_thread, &s, *it));
    }

    for (auto &t : encoder_threads) t.join();
  }
}

int main(int argc, char* argv[]) {
  if (!Hardware::PC()) {
    int ret;
    ret = util::set_realtime_priority(52);
    assert(ret == 0);
    ret = util::set_core_affinity({3});
    assert(ret == 0);
  }
  if (argc > 1) {
    std::string arg1(argv[1]);
    if (arg1 == "--stream") {
      encoderd_thread(stream_cameras_logged);
    } else {
      LOGE("Argument '%s' is not supported", arg1.c_str());
    }
  } else {
    encoderd_thread(cameras_logged);
  }
  return 0;
}
