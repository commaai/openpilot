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

class Barrier {
public:
  explicit Barrier(int count) : count_(count), initial_count_(count) {}

  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (--count_ == 0) {
      count_ = initial_count_;  // Reset count for reuse
      cv_.notify_all();
    } else {
      cv_.wait(lock, [this] { return count_ == initial_count_; });
    }
  }

private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int count_;
  const int initial_count_;
};

std::atomic<uint32_t> start_frame_id(0);

// Handle initial encoder syncing by waiting for all encoders to reach the same frame id
bool sync_encoders(Barrier &barrier, uint32_t frame_id) {
  barrier.wait();
  uint32_t expected = 0;
  std::atomic_compare_exchange_strong(&start_frame_id, &expected, frame_id + 2);
  return frame_id >= start_frame_id.load();
}


void encoder_thread(Barrier &barrier, const LogCameraInfo &cam_info) {
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

      if (!sync_encoders(barrier, extra.frame_id)) {
        continue;
      }
      if (do_exit) break;

      // do rotation if required
      const int frames_per_seg = SEGMENT_LENGTH * MAIN_FPS;
      if (cur_seg >= 0 && extra.frame_id >= ((cur_seg + 1) * frames_per_seg) + start_frame_id) {
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
  Barrier barrier(N);

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
      encoder_threads.push_back(std::thread(encoder_thread, std::ref(barrier), *it));
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
