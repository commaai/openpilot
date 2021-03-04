#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <inttypes.h>
#include <pthread.h>
#include <sys/resource.h>

#include <algorithm>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <random>

#include <ftw.h>

#include "common/timing.h"
#include "common/params.h"
#include "common/swaglog.h"
#include "common/util.h"
#include "camerad/cameras/camera_common.h"
#include "logger.h"
#include "messaging.hpp"
#include "services.h"

#include "visionipc.h"
#include "visionipc_client.h"

#include "encoder.h"
#if defined(QCOM) || defined(QCOM2)
#include "omx_encoder.h"
#define Encoder OmxEncoder
#else
#include "raw_logger.h"
#define Encoder RawLogger
#endif

namespace {

constexpr int MAIN_FPS = 20;

#ifndef QCOM2
#define MAIN_BITRATE 5000000
#define DCAM_BITRATE 2500000
#else
#define MAIN_BITRATE 10000000
#define DCAM_BITRATE MAIN_BITRATE
#endif

#define NO_CAMERA_PATIENCE 500 // fall back to time-based rotation if all cameras are dead

const int SEGMENT_LENGTH = getenv("LOGGERD_TEST") ? atoi(getenv("LOGGERD_SEGMENT_LENGTH")) : 60;

ExitHandler do_exit;

LogCameraInfo cameras_logged[] = {
  { .type = ROAD_CAM,
    .stream_type = VISION_STREAM_YUV_BACK,
    .filename = "fcamera.hevc",
    .frame_packet_name = "roadCameraState",
    .fps = MAIN_FPS,
    .bitrate = MAIN_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = true
  },
#if defined(QCOM) || defined(QCOM2)
  { .type = DRIVER_CAM,
    .stream_type = VISION_STREAM_YUV_FRONT,
    .filename = "dcamera.hevc",
    .frame_packet_name = "driverCameraState",
    .fps = MAIN_FPS, // on EONs, more compressed this way
    .bitrate = DCAM_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = false
  },
#endif
#ifdef QCOM2
  { .type = WIDE_ROAD_CAM,
    .stream_type = VISION_STREAM_YUV_WIDE,
    .filename = "ecamera.hevc",
    .frame_packet_name = "wideRoadCameraState",
    .fps = MAIN_FPS,
    .bitrate = MAIN_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = false
  },
#endif
};

LogCameraInfo qcam_info = {
    .filename = "qcamera.ts",
    .fps = MAIN_FPS,
    .bitrate = 256000,
    .is_h265 = false,
    .downscale = true,
#ifndef QCOM2
    .frame_width = 480, .frame_height = 360
#else
    .frame_width = 526, .frame_height = 330 // keep pixel count the same?
#endif
};

class RotateState {
public:
  SubSocket* fpkt_sock;
  uint32_t stream_frame_id, log_frame_id, last_rotate_frame_id;
  LogCameraInfo cam_info;
  bool should_rotate, initialized;
  std::atomic<bool> rotating;
  std::atomic<int> cur_seg;

  RotateState(SubSocket *sock, const LogCameraInfo& info) : fpkt_sock(sock), cam_info(info), stream_frame_id(0), log_frame_id(0),
                  last_rotate_frame_id(UINT32_MAX), should_rotate(false), initialized(false), rotating(false), cur_seg(-1) {};

  void waitLogThread() {
    std::unique_lock<std::mutex> lk(fid_lock);
    while (stream_frame_id > log_frame_id           // if the log camera is older, wait for it to catch up.
           && (stream_frame_id - log_frame_id) < 8  // but if its too old then there probably was a discontinuity (visiond restarted)
           && !do_exit) {
      cv.wait(lk);
    }
  }

  void cancelWait() {
    cv.notify_one();
  }

  void setStreamFrameId(uint32_t frame_id) {
    fid_lock.lock();
    stream_frame_id = frame_id;
    fid_lock.unlock();
    cv.notify_one();
  }

  void setLogFrameId(uint32_t frame_id) {
    fid_lock.lock();
    log_frame_id = frame_id;
    fid_lock.unlock();
    cv.notify_one();
  }

  void rotate() {
    std::unique_lock<std::mutex> lk(fid_lock);
    should_rotate = true;
    last_rotate_frame_id = stream_frame_id;
  }

  void finish_rotate() {
    std::unique_lock<std::mutex> lk(fid_lock);
    should_rotate = false;
  }

private:
  std::mutex fid_lock;
  std::condition_variable cv;
};

struct LoggerdState {
  Context *ctx;
  LoggerState logger = {};
  char segment_path[4096];
  int rotate_segment;
  pthread_mutex_t rotate_lock;
  std::vector<std::unique_ptr<RotateState>> rotate_states;
};
LoggerdState s;

void encoder_thread(RotateState *rs) {
  LogCameraInfo &cam_info = rs->cam_info;
  set_thread_name(cam_info.filename);

  int cnt = 0;
  LoggerHandle *lh = NULL;
  std::vector<Encoder *> encoders;
  VisionIpcClient vipc_client = VisionIpcClient("camerad", cam_info.stream_type, false);

  while (!do_exit) {
    if (!vipc_client.connect(false)){
      util::sleep_for(100);
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
        encoders.push_back(new Encoder(qcam_info.filename,
                                       qcam_info.frame_width, qcam_info.frame_height,
                                       qcam_info.fps, qcam_info.bitrate, qcam_info.is_h265, qcam_info.downscale));
      }
    }

    while (!do_exit) {
      VisionIpcBufExtra extra;
      VisionBuf* buf = vipc_client.recv(&extra);
      if (buf == nullptr){
        continue;
      }

      //printf("logger latency to tsEof: %f\n", (double)(nanos_since_boot() - extra.timestamp_eof) / 1000000.0);

      // all the rotation stuff
      {
        pthread_mutex_lock(&s.rotate_lock);
        pthread_mutex_unlock(&s.rotate_lock);

        // wait if camera pkt id is older than stream
        rs->waitLogThread();

        if (do_exit) break;

        // rotate the encoder if the logger is on a newer segment
        if (rs->should_rotate) {
          LOGW("camera %d rotate encoder to %s", cam_info.type, s.segment_path);

          if (!rs->initialized) {
            rs->last_rotate_frame_id = extra.frame_id - 1;
            rs->initialized = true;
          }

          // get new logger handle for new segment
          if (lh) {
            lh_close(lh);
          }
          lh = logger_get_handle(&s.logger);

          // wait for all to start rotating
          rs->rotating = true;
          for(auto &r : s.rotate_states) {
             while(!r->rotating && !do_exit) util::sleep_for(5);
          }

          pthread_mutex_lock(&s.rotate_lock);
          for (auto &e : encoders) {
            e->encoder_close();
            e->encoder_open(s.segment_path);
          }
          rs->cur_seg = s.rotate_segment;
          pthread_mutex_unlock(&s.rotate_lock);

          // wait for all to finish rotating
          for(auto &r : s.rotate_states) {
             while(r->cur_seg != s.rotate_segment && !do_exit) util::sleep_for(5);
          }
          rs->rotating = false;
          rs->finish_rotate();
        }
      }

      rs->setStreamFrameId(extra.frame_id);

      // encode a frame
      for (int i = 0; i < encoders.size(); ++i) {
        int out_id = encoders[i]->encode_frame(buf->y, buf->u, buf->v,
                                               buf->width, buf->height, extra.timestamp_eof);
        if (i == 0 && out_id != -1) {
          // publish encode index
          MessageBuilder msg;
          // this is really ugly
          auto eidx = cam_info.type == DRIVER_CAM ? msg.initEvent().initDriverEncodeIdx() :
                     (cam_info.type == WIDE_ROAD_CAM ? msg.initEvent().initWideRoadEncodeIdx() : msg.initEvent().initRoadEncodeIdx());
          eidx.setFrameId(extra.frame_id);
          eidx.setTimestampSof(extra.timestamp_sof);
          eidx.setTimestampEof(extra.timestamp_eof);
    #ifdef QCOM2
          eidx.setType(cereal::EncodeIndex::Type::FULL_H_E_V_C);
    #else
          eidx.setType(cam_info.type == DRIVER_CAM ? cereal::EncodeIndex::Type::FRONT : cereal::EncodeIndex::Type::FULL_H_E_V_C);
    #endif
          eidx.setEncodeId(cnt);
          eidx.setSegmentNum(rs->cur_seg);
          eidx.setSegmentId(out_id);
          if (lh) {
            auto bytes = msg.toBytes();
            lh_log(lh, bytes.begin(), bytes.size(), false);
          }
        }
      }

      cnt++;
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

  setpriority(PRIO_PROCESS, 0, -12);

  clear_locks();

  // setup messaging
  typedef struct QlogState {
    int counter, freq;
  } QlogState;
  std::map<SubSocket*, QlogState> qlog_states;

  s.ctx = Context::create();
  Poller * poller = Poller::create();
  std::vector<SubSocket*> socks;

  // subscribe to all socks
  for (const auto& it : services) {
    if (!it.should_log) continue;

    SubSocket * sock = SubSocket::create(s.ctx, it.name);
    assert(sock != NULL);
    poller->registerSocket(sock);
    socks.push_back(sock);

    auto cam_info = std::find_if(std::begin(cameras_logged), std::end(cameras_logged), [it](auto &ci) { 
      return strcmp(it.name, ci.frame_packet_name) == 0; 
    });
    if (cam_info != std::end(cameras_logged) && (cam_info->type != DRIVER_CAM || Params().read_db_bool("RecordFront"))) {
      s.rotate_states.push_back(std::make_unique<RotateState>(sock, *cam_info));
    }
    qlog_states[sock] = {.counter = 0, .freq = it.decimation};
  }

  // init logger
  logger_init(&s.logger, "rlog", true);

  // init encoders
  pthread_mutex_init(&s.rotate_lock, NULL);

  std::vector<std::thread> encoder_threads;
  for (auto &rs : s.rotate_states) {
    encoder_threads.push_back(std::thread(encoder_thread, rs.get()));
  }

  uint64_t msg_count = 0;
  uint64_t bytes_count = 0;
  kj::Array<capnp::word> buf = kj::heapArray<capnp::word>(1024);

  double start_ts = seconds_since_boot();
  double last_rotate_tms = millis_since_boot();
  double last_camera_seen_tms = millis_since_boot();
  while (!do_exit) {
    // TODO: fix msgs from the first poll getting dropped
    // poll for new messages on all sockets
    for (auto sock : poller->poll(1000)) {

      // drain socket
      Message * last_msg = nullptr;
      while (!do_exit) {
        Message * msg = sock->receive(true);
        if (!msg){
          break;
        }
        delete last_msg;
        last_msg = msg;

        QlogState& qs = qlog_states[sock];
        logger_log(&s.logger, (uint8_t*)msg->getData(), msg->getSize(), qs.counter == 0 && qs.freq != -1);
        if (qs.freq != -1) {
          qs.counter = (qs.counter + 1) % qs.freq;
        }

        bytes_count += msg->getSize();
        if ((++msg_count % 1000) == 0) {
          double ts = seconds_since_boot();
          LOGD("%lu messages, %.2f msg/sec, %.2f KB/sec", msg_count, msg_count * 1.0 / (ts - start_ts), bytes_count * 0.001 / (ts - start_ts));
        }
      }

      if (last_msg) {
        auto it = std::find_if(s.rotate_states.begin(), s.rotate_states.end(), [&](auto &r) {return r->fpkt_sock == sock;});
        if (it != s.rotate_states.end()) {
          // track camera frames to sync to encoder
          // only process last frame
          const uint8_t* data = (uint8_t*)last_msg->getData();
          const size_t len = last_msg->getSize();
          const size_t size = len / sizeof(capnp::word) + 1;
          if (buf.size() < size) {
            buf = kj::heapArray<capnp::word>(size);
          }
          memcpy(buf.begin(), data, len);

          capnp::FlatArrayMessageReader cmsg(buf);
          cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();

          auto &rs = *it;
          if (rs->cam_info.type == ROAD_CAM) {
            rs->setLogFrameId(event.getRoadCameraState().getFrameId());
          } else if (rs->cam_info.type == DRIVER_CAM) {
            rs->setLogFrameId(event.getDriverCameraState().getFrameId());
          } else if (rs->cam_info.type == WIDE_ROAD_CAM) {
            rs->setLogFrameId(event.getWideRoadCameraState().getFrameId());
          }
          last_camera_seen_tms = millis_since_boot();
        }
      }
      delete last_msg;
    }

    bool new_segment = s.logger.part == -1;
    if (s.logger.part > -1) {
      double tms = millis_since_boot();
      if (tms - last_camera_seen_tms <= NO_CAMERA_PATIENCE && encoder_threads.size() > 0) {
        new_segment = true;
        for (auto &r : s.rotate_states) {
          // this *should* be redundant on tici since all camera frames are synced
          new_segment &= (((r->stream_frame_id >= r->last_rotate_frame_id + SEGMENT_LENGTH * MAIN_FPS) &&
                          (!r->should_rotate) && (r->initialized)));
#ifndef QCOM2
          break; // only look at fcamera frame id if not QCOM2
#endif
        }
      } else {
        if (tms - last_rotate_tms > SEGMENT_LENGTH * 1000) {
          new_segment = true;
          LOGW("no camera packet seen. auto rotated");
        }
      }
    }

    // rotate to new segment
    if (new_segment) {
      pthread_mutex_lock(&s.rotate_lock);
      last_rotate_tms = millis_since_boot();

      int err = logger_next(&s.logger, LOG_ROOT.c_str(), s.segment_path, sizeof(s.segment_path), &s.rotate_segment);
      assert(err == 0);
      LOGW((s.logger.part == 0) ? "logging to %s" : "rotated to %s", s.segment_path);

      // rotate encoders
      for (auto &r : s.rotate_states) r->rotate();
      pthread_mutex_unlock(&s.rotate_lock);
    }
  }

  LOGW("closing encoders");
  for (auto &r : s.rotate_states) r->cancelWait();
  for (auto &t : encoder_threads) t.join();

  LOGW("closing logger");
  logger_close(&s.logger);

  if (do_exit.power_failure){
    LOGE("power failure");
    sync();
  }

  // messaging cleanup
  for (auto sock : socks) delete sock;
  delete poller;
  delete s.ctx;

  return 0;
}
