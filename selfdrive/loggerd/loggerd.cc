#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <unistd.h>
#include <signal.h>
#include <inttypes.h>
#include <sys/resource.h>
#include <iostream>
#include <fstream>
#include <streambuf>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <random>

#include <ftw.h>
#ifdef QCOM
#include <cutils/properties.h>
#endif

#include "common/version.h"
#include "common/timing.h"
#include "common/params.h"
#include "common/swaglog.h"
#include "common/visionipc.h"
#include "common/utilpp.h"
#include "common/util.h"
#include "camerad/cameras/camera_common.h"
#include "logger.h"
#include "messaging.hpp"
#include "services.h"

#if !(defined(QCOM) || defined(QCOM2))
// no encoder on PC
#define DISABLE_ENCODER
#endif

#ifndef DISABLE_ENCODER
#include "encoder.h"
#include "raw_logger.h"
#endif

#define MAIN_BITRATE 5000000
#define QCAM_BITRATE 128000
#define MAIN_FPS 20
#ifndef QCOM2
#define MAX_CAM_IDX LOG_CAMERA_ID_DCAMERA
#define DCAM_BITRATE 2500000
#else
#define MAX_CAM_IDX LOG_CAMERA_ID_ECAMERA
#define DCAM_BITRATE MAIN_BITRATE
#endif

#define NO_CAMERA_PATIENCE 500 // fall back to time-based rotation if all cameras are dead

LogCameraInfo cameras_logged[LOG_CAMERA_ID_MAX] = {
  [LOG_CAMERA_ID_FCAMERA] = {
    .stream_type = VISION_STREAM_YUV,
    .filename = "fcamera.hevc",
    .frame_packet_name = "frame",
    .encode_idx_name = "encodeIdx",
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
    .encode_idx_name = "frontEncodeIdx",
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
    .encode_idx_name = "wideEncodeIdx",
    .fps = MAIN_FPS,
    .bitrate = MAIN_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = false
  },
  [LOG_CAMERA_ID_QCAMERA] = {
    .filename = "qcamera.ts",
    .fps = MAIN_FPS,
    .bitrate = QCAM_BITRATE,
    .is_h265 = false,
    .downscale = true,
#ifndef QCOM2
    .frame_width = 480, .frame_height = 360
#else
    .frame_width = 526, .frame_height = 330 // keep pixel count the same?
#endif
  },
};
#define SEGMENT_LENGTH 60

#define LOG_ROOT "/data/media/0/realdata"

#define RAW_CLIP_LENGTH 100 // 5 seconds at 20fps
#define RAW_CLIP_FREQUENCY (randrange(61, 8*60)) // once every ~4 minutes

namespace {

double randrange(double a, double b) __attribute__((unused));
double randrange(double a, double b) {
  static std::mt19937 gen(millis_since_boot());

  std::uniform_real_distribution<> dist(a, b);
  return dist(gen);
}

volatile sig_atomic_t do_exit = 0;
static void set_do_exit(int sig) {
  do_exit = 1;
}

struct LoggerdState {
  Context *ctx;
  LoggerState logger;
  int segment_length;
  char segment_path[4096];
  std::atomic<double> last_rotate_tms;
  
  std::atomic<int> rotate_segment;
  std::mutex rotate_mutex;
  std::condition_variable rotate_cv;
};

#ifndef DISABLE_ENCODER
class Encoder {
public:
 Encoder(LoggerdState* s, int cam_idx, bool raw_clips, const VisionStreamBufs& buf_info)
     : log_s(s), cam_idx(cam_idx), buf_info(buf_info) {
   LOGD("encoder init %dx%d", buf_info.width, buf_info.height);
   const LogCameraInfo& ci = cameras_logged[cam_idx];
   encoder = std::make_unique<EncoderState>();
   encoder_init(encoder.get(), ci.filename, buf_info.width, buf_info.height, ci.fps,
                ci.bitrate, ci.is_h265, ci.downscale);

   if (ci.has_qcamera) {
     const LogCameraInfo& qci = cameras_logged[LOG_CAMERA_ID_QCAMERA];
     encoder_alt = std::make_unique<EncoderState>();
     encoder_init(encoder_alt.get(), qci.filename, qci.frame_width, qci.frame_height, qci.fps,
                  qci.bitrate, qci.is_h265, qci.downscale);
   }
   if (raw_clips) {
     rawlogger_start_time = seconds_since_boot() + RAW_CLIP_FREQUENCY;
     raw_logger = std::make_unique<RawLogger>("prcamera", buf_info.width, buf_info.height, MAIN_FPS);
   }
   idx_sock.reset(PubSocket::create(log_s->ctx, ci.encode_idx_name));
   frame_sock.reset(SubSocket::create(log_s->ctx, ci.frame_packet_name));
   for (const auto& it : services) {
     if (strcmp(it.name, ci.frame_packet_name) == 0) {
       frame_sock_decimation = it.decimation;
     }
   }
 }

  ~Encoder() {
    if (encoder) {
      LOG("encoder destroy");
      encoder_close(encoder.get());
      encoder_destroy(encoder.get());
    }
    if (encoder_alt) {
      LOG("encoder alt destroy");
      encoder_close(encoder_alt.get());
      encoder_destroy(encoder_alt.get());
    }

    if (raw_logger) {
      raw_logger->Close();
    }
    if (lh) {
      lh_close(lh);
    }
  }

  void LogFrame(VIPCBuf* buf, VIPCBufExtra* extra) {
    RotateIfNeeded(extra);

    total_frame_cnt++;
    segment_frame_cnt++;

    // log frame messages
    Message *msg;
    while((msg = frame_sock->receive(true)) != nullptr) {
      lh_log(lh, (uint8_t*)msg->getData(), msg->getSize(),
             frame_sock_decimation == -1 ? false : total_frame_cnt % frame_sock_decimation == 0);
      delete msg;
    }
    
    uint8_t* y = (uint8_t*)buf->addr;
    uint8_t* u = y + (buf_info.width * buf_info.height);
    uint8_t* v = u + (buf_info.width / 2) * (buf_info.height / 2);
    // encode hevc
    int out_id = encoder_encode_frame(encoder.get(), y, u, v, buf_info.width, buf_info.height, extra);
    LogEncodeIdx(extra->frame_id,
                 cam_idx == LOG_CAMERA_ID_DCAMERA ? cereal::EncodeIndex::Type::FRONT : cereal::EncodeIndex::Type ::FULL_H_E_V_C,
                 out_id, idx_sock.get());
    if (encoder_alt) {
      encoder_encode_frame(encoder_alt.get(), y, u, v, buf_info.width, buf_info.height, extra);
    }

    if (raw_logger) {
      if (double ts = seconds_since_boot(); ts > rawlogger_start_time) {
        // encode raw if in clip
        int out_segment = -1;
        int out_id = raw_logger->LogFrame(total_frame_cnt, y, u, v, &out_segment);
        LogEncodeIdx(extra->frame_id, cereal::EncodeIndex::Type::FULL_LOSSLESS_CLIP, out_id);
        if (rawlogger_clip_cnt == 0) {
          LOG("starting raw clip in seg %d", encoder_segment);
        }
        // close rawlogger if clip ended
        rawlogger_clip_cnt++;
        if (rawlogger_clip_cnt >= RAW_CLIP_LENGTH) {
          raw_logger->Close();

          rawlogger_clip_cnt = 0;
          rawlogger_start_time = ts + RAW_CLIP_FREQUENCY;

          LOG("ending raw clip in seg %d, next in %.1f sec", encoder_segment, rawlogger_start_time - ts);
        }
      }
    }
  }

private:
  void RotateIfNeeded(VIPCBufExtra* extra) {
    if ((encoder_segment == -1 || encoder_segment != log_s->rotate_segment) ||
        segment_frame_cnt == log_s->segment_length * MAIN_FPS) {
      char segment_path[4096] = {};
      {  // wait log finished rotated
        std::unique_lock<std::mutex> lock(log_s->rotate_mutex);
        log_s->rotate_cv.wait(lock, [&] { return log_s->rotate_segment > encoder_segment; });
        strcpy(segment_path, log_s->segment_path);
        encoder_segment = log_s->rotate_segment;
      }
      encoder_rotate(encoder.get(), segment_path);
      if (encoder_alt) {
        encoder_rotate(encoder_alt.get(), segment_path);
      }
      if (raw_logger) {
        raw_logger->Rotate(segment_path, encoder_segment);
      }
      if (lh) {
        lh_close(lh);
      }
      lh = logger_get_handle(&log_s->logger);
      segment_frame_cnt = 0;
      LOGW("camera %d rotate encoder to %s. %d", cam_idx, segment_path, segment_frame_cnt);
    }
  }

  void LogEncodeIdx(uint32_t frame_id, cereal::EncodeIndex::Type type, uint32_t segment_id, PubSocket* pub_sock = nullptr) {
    MessageBuilder msg;
    auto eidx = msg.initEvent().initEncodeIdx();
    eidx.setFrameId(frame_id);
    eidx.setType(type);
    eidx.setEncodeId(total_frame_cnt);
    eidx.setSegmentNum(encoder_segment);
    eidx.setSegmentId(segment_id);
    auto bytes = msg.toBytes();
    if (idx_sock && idx_sock->send((char*)bytes.begin(), bytes.size()) < 0) {
      printf("err sending encodeIdx pkt: %s\n", strerror(errno));
    }
    lh_log(lh, bytes.begin(), bytes.size(), false);
  }

  int cam_idx;
  LoggerdState* log_s;
  VisionStreamBufs buf_info;
  std::unique_ptr<EncoderState> encoder, encoder_alt;
  std::unique_ptr<PubSocket> idx_sock;
  std::unique_ptr<SubSocket> frame_sock;
  int frame_sock_decimation = -1;
  std::unique_ptr<RawLogger> raw_logger;
  double rawlogger_start_time = 0;
  int rawlogger_clip_cnt = 0;

  LoggerHandle* lh = nullptr;
  int encoder_segment = -1;
  uint32_t segment_frame_cnt = 0, total_frame_cnt = 0;
};

void encoder_thread(LoggerdState *s, bool raw_clips, int cam_idx) {
  {
    char thread_name[64];
    snprintf(thread_name, sizeof(thread_name), "%sEncoder", cameras_logged[cam_idx].frame_packet_name);
    set_thread_name(thread_name);
  }
  std::unique_ptr<Encoder> encoder;
  VisionStream stream;
  while (!do_exit) {
    VisionStreamBufs buf_info;
    int err = visionstream_init(&stream, cameras_logged[cam_idx].stream_type, false, &buf_info);
    if (err != 0) {
      LOGW("visionstream connect fail");
      usleep(100000);
      continue;
    }
    if (!encoder) {
      encoder = std::make_unique<Encoder>(s, cam_idx, raw_clips, buf_info);
    }
    while (!do_exit) {
      VIPCBufExtra extra;
      VIPCBuf* buf = visionstream_get(&stream, &extra);
      if (buf == NULL) {
        LOGW("visionstream get failed");
        break;
      }
      encoder->LogFrame(buf, &extra);
    }
    visionstream_destroy(&stream);
  }
}
#endif

}

void append_property(const char* key, const char* value, void *cookie) {
  std::vector<std::pair<std::string, std::string> > *properties =
    (std::vector<std::pair<std::string, std::string> > *)cookie;

  properties->push_back(std::make_pair(std::string(key), std::string(value)));
}

kj::Array<capnp::word> gen_init_data() {
  MessageBuilder msg;
  auto init = msg.initEvent().initInitData();

  init.setDeviceType(cereal::InitData::DeviceType::NEO);
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

  const char* clean = getenv("CLEAN");
  if (!clean) {
    init.setDirty(true);
  }
  Params params = Params();

  std::vector<char> git_commit = params.read_db_bytes("GitCommit");
  if (git_commit.size() > 0) {
    init.setGitCommit(capnp::Text::Reader(git_commit.data(), git_commit.size()));
  }

  std::vector<char> git_branch = params.read_db_bytes("GitBranch");
  if (git_branch.size() > 0) {
    init.setGitBranch(capnp::Text::Reader(git_branch.data(), git_branch.size()));
  }

  std::vector<char> git_remote = params.read_db_bytes("GitRemote");
  if (git_remote.size() > 0) {
    init.setGitRemote(capnp::Text::Reader(git_remote.data(), git_remote.size()));
  }

  init.setPassive(params.read_db_bool("Passive"));
  {
    // log params
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
  ftw(LOG_ROOT, clear_locks_fn, 16);
}

static void bootlog(LoggerdState &s) {
  {
    auto words = gen_init_data();
    auto bytes = words.asBytes();
    logger_init(&s.logger, "bootlog", bytes.begin(), bytes.size(), false);
  }

  int err = logger_next(&s.logger, LOG_ROOT, s.segment_path, sizeof(s.segment_path), nullptr);
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

typedef struct QlogState {
  int counter;
  int freq;
} QlogState;


void rotate_if_needed(LoggerdState &s) {
  double tms = millis_since_boot(); 
  if ((tms - s.last_rotate_tms) >= s.segment_length * 1000) {
    {
      std::unique_lock<std::mutex> lock(s.rotate_mutex);
      int segment = 0;
      int err = logger_next(&s.logger, LOG_ROOT, s.segment_path, sizeof(s.segment_path), &segment);
      assert(err == 0);
      s.rotate_segment = segment;
      s.last_rotate_tms = tms;
    }
    s.rotate_cv.notify_all();
    LOGW(s.rotate_segment == 0 ? "logging to %s" :"rotated to %s", s.segment_path);
  }
}

int main(int argc, char** argv) {
  LoggerdState s = {};

  if (argc > 1 && strcmp(argv[1], "--bootlog") == 0) {
    bootlog(s);
    return 0;
  }
  s.segment_length = SEGMENT_LENGTH;
  if (getenv("LOGGERD_TEST")) {
    s.segment_length = atoi(getenv("LOGGERD_SEGMENT_LENGTH"));
  }
  bool record_front = true;
#ifndef QCOM2
  record_front = Params().read_db_bool("RecordFront");
#endif

  setpriority(PRIO_PROCESS, 0, -12);

  clear_locks();

  signal(SIGINT, (sighandler_t)set_do_exit);
  signal(SIGTERM, (sighandler_t)set_do_exit);

  s.ctx = Context::create();
  Poller * poller = Poller::create();

  // subscribe to all services

  std::map<SubSocket*, QlogState> qlog_states;
  for (const auto& it : services) {
    if (it.should_log) {
      std::string name = it.name;
      SubSocket* sock = SubSocket::create(s.ctx, name);
      assert(sock != NULL);
      qlog_states[sock] = {.counter = (it.decimation == -1) ? -1 : 0,
                           .freq = it.decimation};
      for (int cid = 0; cid <= MAX_CAM_IDX; cid++) {
        if (name == cameras_logged[cid].frame_packet_name) {
          continue;
        }
      }
      poller->registerSocket(sock);
      
    }
  }

  {
    auto words = gen_init_data();
    auto bytes = words.asBytes();
    logger_init(&s.logger, "rlog", bytes.begin(), bytes.size(), true);
  }

  std::vector<std::thread> encoder_threads;
#ifndef DISABLE_ENCODER
  // rear camera
  encoder_threads.push_back(std::thread(encoder_thread, &s, false, LOG_CAMERA_ID_FCAMERA));
  // front camera
  if (record_front) {
    encoder_threads.push_back(std::thread(encoder_thread, &s, false, LOG_CAMERA_ID_DCAMERA));
  }
  #ifdef QCOM2
  // wide camera
  std::thread wide_encoder_thread_handle(encoder_thread, &s, false, LOG_CAMERA_ID_ECAMERA);
  s.rotate_state[LOG_CAMERA_ID_ECAMERA].enabled = true;
  #endif
#endif

  uint64_t msg_count = 0;
  uint64_t bytes_count = 0;
  double start_ts = seconds_since_boot();
  while (!do_exit) {
    rotate_if_needed(s);
    
    for (auto sock : poller->poll(10)) {
      QlogState& qs = qlog_states[sock];
      Message* msg;
      while((msg = sock->receive(true)) != nullptr) {
        logger_log(&s.logger, (uint8_t*)msg->getData(), msg->getSize(), qs.counter == 0);
        if (qs.counter != -1) {
          qs.counter = (qs.counter + 1) % qs.freq;
        }

        bytes_count += msg->getSize();
        delete msg;

        if ((++msg_count % 1000) == 0) {
          double ts = seconds_since_boot();
          LOGD("%lu messages, %.2f msg/sec, %.2f KB/sec", msg_count, msg_count * 1.0 / (ts - start_ts), bytes_count * 0.001 / (ts - start_ts));
        }
      }
    }
  }

  LOGW("joining threads");
  for (auto& t : encoder_threads) { t.join(); }
  LOGW("encoder joined");

  logger_close(&s.logger);

  for (auto s : qlog_states) { delete s.first; }

  delete poller;
  delete s.ctx;
  return 0;
}
