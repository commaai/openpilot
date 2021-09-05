#include <ftw.h>
#include <sys/resource.h>
#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <string>
#include <thread>
#include <unordered_map>

#include "cereal/services.h"
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionipc_client.h"
#include "selfdrive/common/params.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"
#include "selfdrive/hardware/hw.h"

#include "selfdrive/loggerd/encoder.h"
#if defined(QCOM) || defined(QCOM2)
#include "selfdrive/loggerd/omx_encoder.h"
#define Encoder OmxEncoder
#else
#include "selfdrive/loggerd/raw_logger.h"
#define Encoder RawLogger
#endif

#include "selfdrive/loggerd/logging.h"

namespace {

constexpr int MAIN_FPS = 20;
const int MAIN_BITRATE = Hardware::TICI() ? 10000000 : 5000000;
const int DCAM_BITRATE = Hardware::TICI() ? MAIN_BITRATE : 2500000;

ExitHandler do_exit;

const LogCameraInfo cameras_logged[] = {
  {
    .type = RoadCam,
    .stream_type = VISION_STREAM_YUV_BACK,
    .filename = "fcamera.hevc",
    .frame_packet_name = "roadCameraState",
    .fps = MAIN_FPS,
    .bitrate = MAIN_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = true,
    .trigger_rotate = true,
    .enable = true,
  },
  {
    .type = DriverCam,
    .stream_type = VISION_STREAM_YUV_FRONT,
    .filename = "dcamera.hevc",
    .frame_packet_name = "driverCameraState",
    .fps = MAIN_FPS, // on EONs, more compressed this way
    .bitrate = DCAM_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = false,
    .trigger_rotate = Hardware::TICI(),
    .enable = !Hardware::PC() && Params().getBool("RecordFront"),
  },
  {
    .type = WideRoadCam,
    .stream_type = VISION_STREAM_YUV_WIDE,
    .filename = "ecamera.hevc",
    .frame_packet_name = "wideRoadCameraState",
    .fps = MAIN_FPS,
    .bitrate = MAIN_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = false,
    .trigger_rotate = true,
    .enable = Hardware::TICI(),
  },
};
const LogCameraInfo qcam_info = {
  .filename = "qcamera.ts",
  .fps = MAIN_FPS,
  .bitrate = 256000,
  .is_h265 = false,
  .downscale = true,
  .frame_width = Hardware::TICI() ? 526 : 480,
  .frame_height = Hardware::TICI() ? 330 : 360 // keep pixel count the same?
};

LoggerdState s;

void encoder_thread(const LogCameraInfo &cam_info) {
  set_thread_name(cam_info.filename);

  int cnt = 0, cur_seg = -1;
  int encode_idx = 0;
  LoggerHandle *lh = NULL;
  std::vector<Encoder *> encoders;
  VisionIpcClient vipc_client = VisionIpcClient("camerad", cam_info.stream_type, false);

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
      encoders.push_back(new Encoder(cam_info.filename, buf_info.width, buf_info.height,
                                     cam_info.fps, cam_info.bitrate, cam_info.is_h265, cam_info.downscale));
      // qcamera encoder
      if (cam_info.has_qcamera) {
        encoders.push_back(new Encoder(qcam_info.filename, qcam_info.frame_width, qcam_info.frame_height,
                                       qcam_info.fps, qcam_info.bitrate, qcam_info.is_h265, qcam_info.downscale));
      }
    }

    while (!do_exit) {
      VisionIpcBufExtra extra;
      VisionBuf* buf = vipc_client.recv(&extra);
      if (buf == nullptr) continue;

      if (cam_info.trigger_rotate && !s.sync_encoders(cam_info.type, extra.frame_id)) continue;

      if (cam_info.trigger_rotate) {
        s.last_camera_seen_tms = millis_since_boot();
      }

      auto segment = s.get_segment(cur_seg, cam_info.trigger_rotate && cnt >= SEGMENT_LENGTH * MAIN_FPS, &do_exit);
      if (!segment) break;

      // rotate the encoder if the logger is on a newer segment
      auto [segment_id, segment_path] = *segment;
      if (segment_id > cur_seg) {
        cur_seg = segment_id;
        cnt = 0;

        LOGW("camera %d rotate encoder to %s", cam_info.type, segment_path.c_str());
        for (auto &e : encoders) {
          e->encoder_close();
          e->encoder_open(segment_path.c_str());
        }
        if (lh) {
          lh_close(lh);
        }
        lh = logger_get_handle(&s.logger);
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
          auto eidx = cam_info.type == DriverCam ? msg.initEvent().initDriverEncodeIdx() :
                     (cam_info.type == WideRoadCam ? msg.initEvent().initWideRoadEncodeIdx() : msg.initEvent().initRoadEncodeIdx());
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
          if (lh) {
            // TODO: this should read cereal/services.h for qlog decimation
            auto bytes = msg.toBytes();
            lh_log(lh, bytes.begin(), bytes.size(), true);
          }
        }
      }

      cnt++;
      encode_idx++;
    }

    if (lh) {
      lh_close(lh);
      lh = NULL;
    }
  }

  LOG("encoder destroy");
  for(auto &e : encoders) {
    e->encoder_close();
    delete e;
  }
}

int clear_locks_fn(const char* fpath, const struct stat *sb, int tyupeflag) {
  const char* dot = strrchr(fpath, '.');
  if (dot && strcmp(dot, ".lock") == 0) {
    unlink(fpath);
  }
  return 0;
}

void clear_locks() {
  ftw(LOG_ROOT.c_str(), clear_locks_fn, 16);
}

} // namespace

int main(int argc, char** argv) {
  setpriority(PRIO_PROCESS, 0, -20);

  clear_locks();

  // setup messaging
  typedef struct QlogState {
    int counter, freq;
  } QlogState;
  std::unordered_map<SubSocket*, QlogState> qlog_states;

  Context *ctx = Context::create();
  Poller * poller = Poller::create();

  // subscribe to all socks
  for (const auto& it : services) {
    if (!it.should_log) continue;

    SubSocket * sock = SubSocket::create(ctx, it.name);
    assert(sock != NULL);
    poller->registerSocket(sock);
    qlog_states[sock] = {.counter = 0, .freq = it.decimation};
  }

  // init logger
  s.init();
  Params().put("CurrentRoute", s.logger.route_name);

  // init encoders
  std::vector<std::thread> encoder_threads;
  for (const auto &ci : cameras_logged) {
    if (ci.enable) {
      encoder_threads.push_back(std::thread(encoder_thread, ci));
      if (ci.trigger_rotate) s.max_waiting++;
    }
  }

  uint64_t msg_count = 0, bytes_count = 0;
  double start_ts = millis_since_boot();
  while (!do_exit) {
    // poll for new messages on all sockets
    for (auto sock : poller->poll(1000)) {
      // drain socket
      QlogState &qs = qlog_states[sock];
      Message *msg = nullptr;
      while (!do_exit && (msg = sock->receive(true))) {
        const bool in_qlog = qs.freq != -1 && (qs.counter++ % qs.freq == 0);
        logger_log(&s.logger, (uint8_t *)msg->getData(), msg->getSize(), in_qlog);
        bytes_count += msg->getSize();
        delete msg;

        s.rotate_if_needed();

        if ((++msg_count % 1000) == 0) {
          double seconds = (millis_since_boot() - start_ts) / 1000.0;
          LOGD("%lu messages, %.2f msg/sec, %.2f KB/sec", msg_count, msg_count / seconds, bytes_count * 0.001 / seconds);
        }
      }
    }
  }

  LOGW("closing encoders");
  s.close(&do_exit);
  for (auto &t : encoder_threads) t.join();

  if (do_exit.power_failure) {
    LOGE("power failure");
    sync();
    LOGE("sync done");
  }

  // messaging cleanup
  for (auto &[sock, qs] : qlog_states) delete sock;
  delete poller;
  delete ctx;

  return 0;
}
