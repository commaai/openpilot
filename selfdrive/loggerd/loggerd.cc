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

const bool IS_TICI = util::file_exists("/TICI");
const bool IS_EON = util::file_exists("/EON");
const int SEGMENT_LENGTH = getenv("LOGGERD_TEST") ? atoi(getenv("LOGGERD_SEGMENT_LENGTH")) : 60;

const int NO_CAMERA_PATIENCE = 500; // fall back to time-based rotation if all cameras are dead
const int MAIN_FPS = 20;
const int MAIN_BITRATE = IS_TICI ? 10000000 : 5000000;
const int DCAM_BITRATE = IS_TICI ? 10000000 : 2500000;

ExitHandler do_exit;

typedef cereal::EncodeIndex::Builder (cereal::Event::Builder::*initEncodeIndexFunc)();
typedef cereal::FrameData::Reader (cereal::Event::Reader::*getFrameDataFunc)() const;
typedef struct LogCameraInfo {
  const char* filename;
  const char* frame_packet_name;
  const char* encode_idx_name;
  VisionStreamType stream_type;
  int frame_width, frame_height;
  int fps;
  int bitrate;
  bool is_h265;
  bool downscale;
  bool has_qcamera;
  bool enabled;
  cereal::EncodeIndex::Type edix_type;
  initEncodeIndexFunc initEncoderIndex;
  getFrameDataFunc getFrameData;
} LogCameraInfo;

LogCameraInfo cameras_logged[] = {
  {
    .stream_type = VISION_STREAM_YUV_BACK,
    .filename = "fcamera.hevc",
    .frame_packet_name = "roadCameraState",
    .fps = MAIN_FPS,
    .bitrate = MAIN_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = true,
    .enabled = true,
    .edix_type = cereal::EncodeIndex::Type::FULL_H_E_V_C,
    .initEncoderIndex = &cereal::Event::Builder::initRoadEncodeIdx,
    .getFrameData = &cereal::Event::Reader::getRoadCameraState,
  },
  {
    .stream_type = VISION_STREAM_YUV_FRONT,
    .filename = "dcamera.hevc",
    .frame_packet_name = "driverCameraState",
    .fps = MAIN_FPS, // on EONs, more compressed this way
    .bitrate = DCAM_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = false,
    .enabled = (IS_TICI || IS_EON) && Params().getBool("RecordFront"),
    .edix_type = IS_TICI ? cereal::EncodeIndex::Type::FULL_H_E_V_C : cereal::EncodeIndex::Type::FRONT,
    .initEncoderIndex = &cereal::Event::Builder::initDriverEncodeIdx,
    .getFrameData = &cereal::Event::Reader::getWideRoadCameraState,
  },
  {
    .stream_type = VISION_STREAM_YUV_WIDE,
    .filename = "ecamera.hevc",
    .frame_packet_name = "wideRoadCameraState",
    .fps = MAIN_FPS,
    .bitrate = MAIN_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = false,
    .enabled = IS_TICI,
    .edix_type = cereal::EncodeIndex::Type::FULL_H_E_V_C,
    .initEncoderIndex = &cereal::Event::Builder::initWideRoadEncodeIdx,
    .getFrameData = &cereal::Event::Reader::getDriverCameraState,
  },
};

const LogCameraInfo qcam_info = {
    .filename = "qcamera.ts",
    .fps = MAIN_FPS,
    .bitrate = 256000,
    .is_h265 = false,
    .downscale = true,
    .frame_width = IS_TICI ? 526 : 480,
    .frame_height =IS_TICI ?  330 : 360, // keep pixel count the same?
};

class RotateState {
public:
  SubSocket* fpkt_sock = nullptr;
  uint32_t stream_frame_id = 0, log_frame_id = 0, last_rotate_frame_id = UINT32_MAX;
  bool should_rotate = false, initialized = false;
  std::atomic<bool> rotating = false;
  std::atomic<int> cur_seg = -1;
  const LogCameraInfo *ci;

  RotateState(const LogCameraInfo *cam_info) : ci(cam_info) {};

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
  std::vector<RotateState *> rotate_state;
};
LoggerdState s;

void encoder_thread(int cam_idx, RotateState &rotate_state) {
  const LogCameraInfo &cam_info = *rotate_state.ci;
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
        rotate_state.waitLogThread();

        if (do_exit) break;

        // rotate the encoder if the logger is on a newer segment
        if (rotate_state.should_rotate) {
          LOGW("camera %d rotate encoder to %s", cam_idx, s.segment_path);

          if (!rotate_state.initialized) {
            rotate_state.last_rotate_frame_id = extra.frame_id - 1;
            rotate_state.initialized = true;
          }

          // get new logger handle for new segment
          if (lh) {
            lh_close(lh);
          }
          lh = logger_get_handle(&s.logger);

          // wait for all to start rotating
          rotate_state.rotating = true;
          for(auto &r : s.rotate_state) {
             while(!r->rotating && !do_exit) util::sleep_for(5);
          }

          pthread_mutex_lock(&s.rotate_lock);
          for (auto &e : encoders) {
            e->encoder_close();
            e->encoder_open(s.segment_path);
          }
          rotate_state.cur_seg = s.rotate_segment;
          pthread_mutex_unlock(&s.rotate_lock);

          // wait for all to finish rotating
          for(auto &r : s.rotate_state) {
             while(r->cur_seg != s.rotate_segment && !do_exit) util::sleep_for(5);
          }
          rotate_state.rotating = false;
          rotate_state.finish_rotate();
        }
      }

      rotate_state.setStreamFrameId(extra.frame_id);

      // encode a frame
      for (int i = 0; i < encoders.size(); ++i) {
        int out_id = encoders[i]->encode_frame(buf->y, buf->u, buf->v,
                                               buf->width, buf->height, extra.timestamp_eof);
        if (i == 0 && out_id != -1) {
          // publish encode index
          MessageBuilder msg;
          auto eidx = (msg.initEvent().*cam_info.initEncoderIndex)();
          eidx.setType(cam_info.edix_type);
          eidx.setFrameId(extra.frame_id);
          eidx.setTimestampSof(extra.timestamp_sof);
          eidx.setTimestampEof(extra.timestamp_eof);
          eidx.setEncodeId(cnt);
          eidx.setSegmentNum(rotate_state.cur_seg);
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

  // init rotate stats
  for (const auto &ci : cameras_logged) {
    if (ci.enabled) s.rotate_state.push_back(new RotateState(&ci));
  }

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

    for (auto& r : s.rotate_state) {
      if (strcmp(it.name, r->ci->frame_packet_name) == 0) {
        r->fpkt_sock = sock;
      }
    }
    qlog_states[sock] = {.counter = 0, .freq = it.decimation};
  }

  // init logger
  logger_init(&s.logger, "rlog", true);

  // init encoders
  pthread_mutex_init(&s.rotate_lock, NULL);

  std::vector<std::thread> encoder_threads;
  for (int idx = 0; idx < s.rotate_state.size(); ++idx) {
    encoder_threads.push_back(std::thread(encoder_thread, idx, std::ref(*s.rotate_state[idx])));
  }

  uint64_t msg_count = 0;
  uint64_t bytes_count = 0;
  AlignedBuffer aligned_buf;

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
        auto it = std::find_if(s.rotate_state.begin(), s.rotate_state.end(), [=] (const RotateState *r){
          return sock == r->fpkt_sock;
        });
        if (it != s.rotate_state.end()) {
          // track camera frames to sync to encoder
          // only process last frame
          RotateState *r = *it; 
          capnp::FlatArrayMessageReader cmsg(aligned_buf.align(last_msg));
          r->setLogFrameId((cmsg.getRoot<cereal::Event>().*r->ci->getFrameData)().getFrameId());
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
        for (auto &r : s.rotate_state) {
          // this *should* be redundant on tici since all camera frames are synced
          new_segment &= (r->stream_frame_id >= r->last_rotate_frame_id + SEGMENT_LENGTH * MAIN_FPS) &&
                          !r->should_rotate && r->initialized;
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
      for (auto &r : s.rotate_state) r->rotate();
      pthread_mutex_unlock(&s.rotate_lock);
    }
  }

  LOGW("closing encoders");
  for (auto &r : s.rotate_state) r->cancelWait();
  for (auto &t : encoder_threads) t.join();
  for (auto &r : s.rotate_state) delete r;

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
