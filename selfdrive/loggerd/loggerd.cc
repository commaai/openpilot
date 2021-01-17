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
#include <iostream>
#include <fstream>
#include <streambuf>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <random>

#include <ftw.h>
#ifdef QCOM
#include <cutils/properties.h>
#endif

#include "common/version.h"
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

constexpr int MAIN_BITRATE = 5000000;
constexpr int MAIN_FPS = 20;
#ifndef QCOM2
constexpr int MAX_CAM_IDX = LOG_CAMERA_ID_DCAMERA;
constexpr int DCAM_BITRATE = 2500000;
#else
constexpr int MAX_CAM_IDX = LOG_CAMERA_ID_ECAMERA;
constexpr int DCAM_BITRATE = MAIN_BITRATE;
#endif

#define NO_CAMERA_PATIENCE 500 // fall back to time-based rotation if all cameras are dead

LogCameraInfo cameras_logged[LOG_CAMERA_ID_MAX] = {
  [LOG_CAMERA_ID_FCAMERA] = {
    .stream_type = VISION_STREAM_YUV_BACK,
    .filename = "fcamera.hevc",
    .frame_packet_name = "frame",
    .fps = MAIN_FPS,
    .bitrate = MAIN_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = true
  },
  [LOG_CAMERA_ID_DCAMERA] = {
    .stream_type = VISION_STREAM_YUV_FRONT,
    .filename = "dcamera.hevc",
    .frame_packet_name = "frontFrame",
    .fps = MAIN_FPS, // on EONs, more compressed this way
    .bitrate = DCAM_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = false
  },
  [LOG_CAMERA_ID_ECAMERA] = {
    .stream_type = VISION_STREAM_YUV_WIDE,
    .filename = "ecamera.hevc",
    .frame_packet_name = "wideFrame",
    .fps = MAIN_FPS,
    .bitrate = MAIN_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = false
  },
  [LOG_CAMERA_ID_QCAMERA] = {
    .filename = "qcamera.ts",
    .fps = MAIN_FPS,
    .bitrate = 128000,
    .is_h265 = false,
    .downscale = true,
#ifndef QCOM2
    .frame_width = 480, .frame_height = 360
#else
    .frame_width = 526, .frame_height = 330 // keep pixel count the same?
#endif
  },
};


namespace {

constexpr int SEGMENT_LENGTH = 60;
#if defined(QCOM) || defined(QCOM2)
std::string LOG_ROOT = "/data/media/0/realdata";
#else
std::string LOG_ROOT = util::getenv_default("HOME", "/.comma/media/0/realdata", "/data/media/0/realdata");
#endif

double randrange(double a, double b) __attribute__((unused));
double randrange(double a, double b) {
  static std::mt19937 gen(millis_since_boot());

  std::uniform_real_distribution<> dist(a, b);
  return dist(gen);
}

ExitHandler do_exit;

static bool file_exists(const std::string& fn) {
  std::ifstream f(fn);
  return f.good();
}

class RotateState {
public:
  SubSocket* fpkt_sock;
  uint32_t stream_frame_id, log_frame_id, last_rotate_frame_id;
  bool enabled, should_rotate, initialized;

  RotateState() : fpkt_sock(nullptr), stream_frame_id(0), log_frame_id(0),
                  last_rotate_frame_id(UINT32_MAX), enabled(false), should_rotate(false), initialized(false) {};

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
    if (enabled) {
      std::unique_lock<std::mutex> lk(fid_lock);
      should_rotate = true;
      last_rotate_frame_id = stream_frame_id;
    }
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
  LoggerState logger;
  char segment_path[4096];
  int rotate_segment;
  pthread_mutex_t rotate_lock;
  RotateState rotate_state[LOG_CAMERA_ID_MAX-1];
};
LoggerdState s;

void encoder_thread(int cam_idx) {
  assert(cam_idx < LOG_CAMERA_ID_MAX-1);

  LogCameraInfo &cam_info = cameras_logged[cam_idx];
  RotateState &rotate_state = s.rotate_state[cam_idx];

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
        LogCameraInfo &qcam_info = cameras_logged[LOG_CAMERA_ID_QCAMERA];
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

          if (lh) {
            lh_close(lh);
          }
          lh = logger_get_handle(&s.logger);

          pthread_mutex_lock(&s.rotate_lock);
          for (auto &e : encoders) {
            e->encoder_close();
            e->encoder_open(s.segment_path, s.rotate_segment);
          }
          pthread_mutex_unlock(&s.rotate_lock);
          rotate_state.finish_rotate();
        }
      }

      rotate_state.setStreamFrameId(extra.frame_id);

      // encode a frame
      {
        int out_segment = -1;
        int out_id = encoders[0]->encode_frame(buf->y, buf->u, buf->v,
                                               buf->width, buf->height,
                                               &out_segment, &extra);
        if (encoders.size() > 1) {
          int out_segment_alt = -1;
          encoders[1]->encode_frame(buf->y, buf->u, buf->v,
                                    buf->width, buf->height,
                                    &out_segment_alt, &extra);
        }

        // publish encode index
        MessageBuilder msg;
        // this is really ugly
        auto eidx = cam_idx == LOG_CAMERA_ID_DCAMERA ? msg.initEvent().initFrontEncodeIdx() :
                    (cam_idx == LOG_CAMERA_ID_ECAMERA ? msg.initEvent().initWideEncodeIdx() : msg.initEvent().initEncodeIdx());
        eidx.setFrameId(extra.frame_id);
        eidx.setTimestampSof(extra.timestamp_sof);
        eidx.setTimestampEof(extra.timestamp_eof);
  #ifdef QCOM2
        eidx.setType(cereal::EncodeIndex::Type::FULL_H_E_V_C);
  #else
        eidx.setType(cam_idx == LOG_CAMERA_ID_DCAMERA ? cereal::EncodeIndex::Type::FRONT : cereal::EncodeIndex::Type::FULL_H_E_V_C);
  #endif
        eidx.setEncodeId(cnt);
        eidx.setSegmentNum(out_segment);
        eidx.setSegmentId(out_id);

        if (lh) {
          auto bytes = msg.toBytes();
          lh_log(lh, bytes.begin(), bytes.size(), false);
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

}

void append_property(const char* key, const char* value, void *cookie) {
  std::vector<std::pair<std::string, std::string> > *properties =
    (std::vector<std::pair<std::string, std::string> > *)cookie;

  properties->push_back(std::make_pair(std::string(key), std::string(value)));
}

kj::Array<capnp::word> gen_init_data() {
  MessageBuilder msg;
  auto init = msg.initEvent().initInitData();

  if (file_exists("/EON")) {
    init.setDeviceType(cereal::InitData::DeviceType::NEO);
  } else if (file_exists("/TICI")) {
    init.setDeviceType(cereal::InitData::DeviceType::TICI);
  } else {
    init.setDeviceType(cereal::InitData::DeviceType::PC);
  }

  init.setVersion(capnp::Text::Reader(COMMA_VERSION));

  std::ifstream cmdline_stream("/proc/cmdline");
  std::vector<std::string> kernel_args;
  std::string buf;
  while (cmdline_stream >> buf) {
    kernel_args.push_back(buf);
  }

  auto lkernel_args = init.initKernelArgs(kernel_args.size());
  for (int i=0; i<kernel_args.size(); i++) {
    lkernel_args.set(i, kernel_args[i]);
  }

  init.setKernelVersion(util::read_file("/proc/version"));

#ifdef QCOM
  {
    std::vector<std::pair<std::string, std::string> > properties;
    property_list(append_property, (void*)&properties);

    auto lentries = init.initAndroidProperties().initEntries(properties.size());
    for (int i=0; i<properties.size(); i++) {
      auto lentry = lentries[i];
      lentry.setKey(properties[i].first);
      lentry.setValue(properties[i].second);
    }
  }
#endif

  const char* dongle_id = getenv("DONGLE_ID");
  if (dongle_id) {
    init.setDongleId(std::string(dongle_id));
  }
  init.setDirty(!getenv("CLEAN"));

  // log params
  Params params = Params();
  init.setGitCommit(params.get("GitCommit"));
  init.setGitBranch(params.get("GitBranch"));
  init.setGitRemote(params.get("GitRemote"));
  init.setPassive(params.read_db_bool("Passive"));
  {
    std::map<std::string, std::string> params_map;
    params.read_db_all(&params_map);
    auto lparams = init.initParams().initEntries(params_map.size());
    int i = 0;
    for (auto& kv : params_map) {
      auto lentry = lparams[i];
      lentry.setKey(kv.first);
      lentry.setValue(kv.second);
      i++;
    }
  }
  return capnp::messageToFlatArray(msg);
}

static int clear_locks_fn(const char* fpath, const struct stat *sb, int tyupeflag) {
  const char* dot = strrchr(fpath, '.');
  if (dot && strcmp(dot, ".lock") == 0) {
    unlink(fpath);
  }
  return 0;
}

static void clear_locks() {
  ftw(LOG_ROOT.c_str(), clear_locks_fn, 16);
}

static void bootlog() {
  int err;

  {
    auto words = gen_init_data();
    auto bytes = words.asBytes();
    logger_init(&s.logger, "bootlog", bytes.begin(), bytes.size(), false);
  }

  err = logger_next(&s.logger, LOG_ROOT.c_str(), s.segment_path, sizeof(s.segment_path), &s.rotate_segment);
  assert(err == 0);
  LOGW("bootlog to %s", s.segment_path);

  {
    MessageBuilder msg;
    auto boot = msg.initEvent().initBoot();

    boot.setWallTimeNanos(nanos_since_epoch());

    std::string lastKmsg = util::read_file("/sys/fs/pstore/console-ramoops");
    boot.setLastKmsg(capnp::Data::Reader((const kj::byte*)lastKmsg.data(), lastKmsg.size()));

    std::string lastPmsg = util::read_file("/sys/fs/pstore/pmsg-ramoops-0");
    boot.setLastPmsg(capnp::Data::Reader((const kj::byte*)lastPmsg.data(), lastPmsg.size()));

    std::string launchLog = util::read_file("/tmp/launch_log");
    boot.setLaunchLog(capnp::Text::Reader(launchLog.data(), launchLog.size()));

    auto bytes = msg.toBytes();
    logger_log(&s.logger, bytes.begin(), bytes.size(), false);
  }

  logger_close(&s.logger);
}

int main(int argc, char** argv) {

  setpriority(PRIO_PROCESS, 0, -12);

  if (argc > 1 && strcmp(argv[1], "--bootlog") == 0) {
    bootlog();
    return 0;
  }

  int segment_length = SEGMENT_LENGTH;
  if (getenv("LOGGERD_TEST")) {
    segment_length = atoi(getenv("LOGGERD_SEGMENT_LENGTH"));
  }

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

    for (int cid=0; cid<=MAX_CAM_IDX; cid++) {
      if (std::string(it.name) == cameras_logged[cid].frame_packet_name) {
        s.rotate_state[cid].fpkt_sock = sock;
      }
    }
    qlog_states[sock] = {.counter = 0, .freq = it.decimation};
  }

  // init logger
  {
    auto words = gen_init_data();
    auto bytes = words.asBytes();
    logger_init(&s.logger, "rlog", bytes.begin(), bytes.size(), true);
  }

  // init encoders
  pthread_mutex_init(&s.rotate_lock, NULL);

  // TODO: create these threads dynamically on frame packet presence
  std::vector<std::thread> encoder_threads;
  encoder_threads.push_back(std::thread(encoder_thread, LOG_CAMERA_ID_FCAMERA));
  s.rotate_state[LOG_CAMERA_ID_FCAMERA].enabled = true;

#if defined(QCOM) || defined(QCOM2)
  bool record_front = Params().read_db_bool("RecordFront");
  if (record_front) {
    encoder_threads.push_back(std::thread(encoder_thread, LOG_CAMERA_ID_DCAMERA));
    s.rotate_state[LOG_CAMERA_ID_DCAMERA].enabled = true;
  }

#ifdef QCOM2
  encoder_threads.push_back(std::thread(encoder_thread, LOG_CAMERA_ID_ECAMERA));
  s.rotate_state[LOG_CAMERA_ID_ECAMERA].enabled = true;
#endif
#endif

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
        msg_count++;
      }

      if (last_msg) {
        int fpkt_id = -1;
        for (int cid = 0; cid <=MAX_CAM_IDX; cid++) {
          if (sock == s.rotate_state[cid].fpkt_sock) {
            fpkt_id=cid;
            break;
          }
        }
        if (fpkt_id >= 0) {
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

          if (fpkt_id == LOG_CAMERA_ID_FCAMERA) {
            s.rotate_state[fpkt_id].setLogFrameId(event.getFrame().getFrameId());
          } else if (fpkt_id == LOG_CAMERA_ID_DCAMERA) {
            s.rotate_state[fpkt_id].setLogFrameId(event.getFrontFrame().getFrameId());
          } else if (fpkt_id == LOG_CAMERA_ID_ECAMERA) {
            s.rotate_state[fpkt_id].setLogFrameId(event.getWideFrame().getFrameId());
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
        for (auto &r : s.rotate_state) {
          // this *should* be redundant on tici since all camera frames are synced
          new_segment &= (((r.stream_frame_id >= r.last_rotate_frame_id + segment_length * MAIN_FPS) &&
                          (!r.should_rotate) && (r.initialized)) ||
                          (!r.enabled));
#ifndef QCOM2
          break; // only look at fcamera frame id if not QCOM2
#endif
        }
      } else {
        if (tms - last_rotate_tms > segment_length * 1000) {
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
      for (auto &r : s.rotate_state) r.rotate();
      pthread_mutex_unlock(&s.rotate_lock);
    }

    if ((msg_count % 1000) == 0) {
      double ts = seconds_since_boot();
      LOGD("%lu messages, %.2f msg/sec, %.2f KB/sec", msg_count, msg_count*1.0/(ts-start_ts), bytes_count*0.001/(ts-start_ts));
    }
  }

  LOGW("closing encoders");
  for (auto &r : s.rotate_state) r.cancelWait();
  for (auto &t : encoder_threads) t.join();

  LOGW("closing logger");
  logger_close(&s.logger);

  // messaging cleanup
  for (auto sock : socks) delete sock;
  delete poller;
  delete s.ctx;

  return 0;
}
