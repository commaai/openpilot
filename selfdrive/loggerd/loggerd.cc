#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <poll.h>
#include <inttypes.h>
#include <libyuv.h>
#include <sys/resource.h>
#include <pthread.h>
#include <iostream>
#include <fstream>
#include <streambuf>
#include <thread>
#include <mutex>
#include <condition_variable>
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
const bool is_qcom2 = false;
#define DCAM_BITRATE 2500000
#else
const bool is_qcom2 = true;
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
  int num_encoder_need_sync;

  double last_rotate_tms;
  int num_encoder_rotated;
  std::atomic<int> num_rotating;
  std::atomic<int> rotate_segment;
  std::atomic<double> last_camera_seen_tms;
  std::mutex rotate_lock;
  std::condition_variable rotate_cv;
};
LoggerdState s = {};

#ifndef DISABLE_ENCODER

void encoder_thread(bool raw_clips, int cam_idx) {
  {
    char thread_name[64];
    snprintf(thread_name, sizeof(thread_name), "%sEncoder", cameras_logged[cam_idx].frame_packet_name);
    set_thread_name(thread_name);
  }

  bool encoder_inited = false;
  EncoderState encoder;
  EncoderState encoder_alt;
  bool has_encoder_alt = cameras_logged[cam_idx].has_qcamera;

  std::unique_ptr<PubSocket> idx_sock(PubSocket::create(s.ctx, cameras_logged[cam_idx].encode_idx_name));
  assert(idx_sock != nullptr);
  std::unique_ptr<SubSocket> frame_sock(SubSocket::create(s.ctx, cameras_logged[cam_idx].frame_packet_name));
  assert(frame_sock != nullptr);

  LoggerHandle* lh = nullptr;
  int encoder_segment = -1;
  uint64_t cnt = 0;
  uint32_t segment_encoded_frames = 0;
  char segment_path[4096];
  const bool is_rotate_camera = is_qcom2 ? true : cam_idx == LOG_CAMERA_ID_FCAMERA;
  VisionStream stream;

  while (!do_exit) {
    VisionStreamBufs buf_info;
    int err = visionstream_init(&stream, cameras_logged[cam_idx].stream_type, false, &buf_info);
    if (err != 0) {
      LOGD("visionstream connect fail");
      usleep(100000);
      continue;
    }

    if (!encoder_inited) {
      LOGD("encoder init %dx%d", buf_info.width, buf_info.height);
      encoder_init(&encoder, &cameras_logged[cam_idx], buf_info.width, buf_info.height);

      if (has_encoder_alt) {
        const LogCameraInfo &cam_info = cameras_logged[LOG_CAMERA_ID_QCAMERA];
        encoder_init(&encoder_alt, &cam_info, cam_info.frame_width, cam_info.frame_height);
      }

      encoder_inited = true;
    }

    // dont log a raw clip in the first minute
    double rawlogger_start_time = seconds_since_boot()+RAW_CLIP_FREQUENCY;
    int rawlogger_clip_cnt = 0;
    RawLogger *rawlogger = NULL;
    if (raw_clips) {
      rawlogger = new RawLogger("prcamera", buf_info.width, buf_info.height, MAIN_FPS);
    }

    while (!do_exit) {
      VIPCBufExtra extra;
      VIPCBuf* buf = visionstream_get(&stream, &extra);
      if (buf == NULL) {
        LOG("visionstream get failed");
        break;
      }
      s.last_camera_seen_tms = millis_since_boot();

      bool should_rotate = false;
      const int log_segment = s.rotate_segment;
      if (encoder_segment != log_segment) {
        // rotate if the logger is on a newer segment.
        should_rotate = true;
        encoder_segment = log_segment;
      } else if (is_rotate_camera && segment_encoded_frames == (s.segment_length * MAIN_FPS)) {
        should_rotate = true;
        encoder_segment = log_segment + 1;
        
        // inform the logger that we are rotating
        s.num_rotating += 1;
      }

      if (should_rotate) {
        err = logger_mk_segment_path(&s.logger, LOG_ROOT, encoder_segment, segment_path, sizeof(segment_path), true);
        assert(err == 0);
        encoder_rotate(&encoder, segment_path);
        if (has_encoder_alt) {
          encoder_rotate(&encoder_alt, segment_path);
        }
        if (raw_clips) {
          rawlogger->Rotate(segment_path, encoder_segment);
        }
        if (lh) {
          lh_close(lh);
        }
        segment_encoded_frames = 0;

        if (is_rotate_camera) {
          { // wait for all rotations(encoders&logger) to complete.
            std::unique_lock<std::mutex> lk(s.rotate_lock);
            s.num_encoder_rotated += 1;
            s.rotate_cv.wait(lk, [&] { return (s.num_encoder_rotated == s.num_encoder_need_sync || s.num_encoder_rotated == 0) &&
                                              s.num_rotating == 0; });
            s.num_encoder_rotated = 0;
          }
          s.rotate_cv.notify_all();
        }

        lh = logger_get_handle(&s.logger);
        LOGW("camera %d rotate encoder to %s.", cam_idx, segment_path);
      }

      // log frame messages
      while (!do_exit) {
        Message* msg = frame_sock->receive(true);
        if (!msg) break;

        if (lh) {
          lh_log(lh, (uint8_t*)msg->getData(), msg->getSize(), false);
        }
        delete msg;
      }

      uint8_t *y = (uint8_t*)buf->addr;
      uint8_t *u = y + (buf_info.width*buf_info.height);
      uint8_t *v = u + (buf_info.width/2)*(buf_info.height/2);
      {
        // encode hevc
        int out_id = encoder_encode_frame(&encoder, y, u, v, buf_info.width, buf_info.height, &extra);
        if (has_encoder_alt) {
          encoder_encode_frame(&encoder_alt, y, u, v, buf_info.width, buf_info.height, &extra);
        }

        // publish encode index
        MessageBuilder msg;
        auto eidx = msg.initEvent().initEncodeIdx();
        eidx.setFrameId(extra.frame_id);
        eidx.setType(!is_qcom2 && cam_idx == LOG_CAMERA_ID_DCAMERA ? cereal::EncodeIndex::Type::FRONT : cereal::EncodeIndex::Type::FULL_H_E_V_C);
        eidx.setEncodeId(cnt);
        eidx.setSegmentNum(encoder_segment);
        eidx.setSegmentId(out_id);

        auto bytes = msg.toBytes();

        if (idx_sock->send((char*)bytes.begin(), bytes.size()) < 0) {
          printf("err sending encodeIdx pkt: %s\n", strerror(errno));
        }
        if (lh) {
          lh_log(lh, bytes.begin(), bytes.size(), false);
        }
      }

      if (raw_clips) {
        double ts = seconds_since_boot();
        if (ts > rawlogger_start_time) {
          // encode raw if in clip
          int out_segment = -1;
          int out_id = rawlogger->LogFrame(cnt, y, u, v, &out_segment);

          if (rawlogger_clip_cnt == 0) {
            LOG("starting raw clip in seg %d", out_segment);
          }

          // publish encode index
          MessageBuilder msg;
          auto eidx = msg.initEvent().initEncodeIdx();
          eidx.setFrameId(extra.frame_id);
          eidx.setType(cereal::EncodeIndex::Type::FULL_LOSSLESS_CLIP);
          eidx.setEncodeId(cnt);
          eidx.setSegmentNum(out_segment);
          eidx.setSegmentId(out_id);

          auto bytes = msg.toBytes();
          if (lh) {
            lh_log(lh, bytes.begin(), bytes.size(), false);
          }

          // close rawlogger if clip ended
          rawlogger_clip_cnt++;
          if (rawlogger_clip_cnt >= RAW_CLIP_LENGTH) {
            rawlogger->Close();

            rawlogger_clip_cnt = 0;
            rawlogger_start_time = ts+RAW_CLIP_FREQUENCY;

            LOG("ending raw clip in seg %d, next in %.1f sec", out_segment, rawlogger_start_time-ts);
          }
        }
      }

      cnt++;
      segment_encoded_frames++;
    }

    if (raw_clips) {
      rawlogger->Close();
      delete rawlogger;
    }

    visionstream_destroy(&stream);
  }

  if (encoder_inited) {
    LOG("encoder destroy");
    encoder_close(&encoder);
    encoder_destroy(&encoder);
  }

  if (has_encoder_alt) {
    LOG("encoder alt destroy");
    encoder_close(&encoder_alt);
    encoder_destroy(&encoder_alt);
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

static void bootlog() {
  int err;

  {
    auto words = gen_init_data();
    auto bytes = words.asBytes();
    logger_init(&s.logger, "bootlog", bytes.begin(), bytes.size(), false);
  }

  std::string segment_path;
  err = logger_next(&s.logger, LOG_ROOT, &segment_path);
  assert(err == 0);
  LOGW("bootlog to %s", segment_path.c_str());

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

static void rotate_log_if_needed(LoggerdState *s) {
  bool should_rotate = false;
  if (s->num_encoder_need_sync > 0 && s->num_rotating == s->num_encoder_need_sync) { // all encoders are rotating.
    should_rotate = true;
  } else { // time-based rotation if all cameras are dead
    double tms = millis_since_boot();
    if ((tms - s->last_rotate_tms) > s->segment_length * 1000 &&
        (tms - s->last_camera_seen_tms) > NO_CAMERA_PATIENCE &&
        s->num_rotating == 0) {
      should_rotate = true;
    }
  }
  if (should_rotate) {
    std::string segment_path;
    assert(logger_next(&s->logger, LOG_ROOT, &segment_path, &s->rotate_segment) == 0);
    s->last_rotate_tms = millis_since_boot();
    LOGW("rotated to %s", segment_path.c_str());
    // notify encoders the logger has rotated.
    s->num_rotating = 0;
    s->rotate_cv.notify_all();
  }
}

int main(int argc, char** argv) {
  if (argc > 1 && strcmp(argv[1], "--bootlog") == 0) {
    bootlog();
    return 0;
  }

  s.segment_length = SEGMENT_LENGTH;
  if (getenv("LOGGERD_TEST")) {
    s.segment_length = atoi(getenv("LOGGERD_SEGMENT_LENGTH"));
  }
  bool record_front = is_qcom2 ? true : Params().read_db_bool("RecordFront");

  setpriority(PRIO_PROCESS, 0, -12);

  clear_locks();

  signal(SIGINT, (sighandler_t)set_do_exit);
  signal(SIGTERM, (sighandler_t)set_do_exit);

  s.ctx = Context::create();
  Poller * poller = Poller::create();

  // subscribe to all services

  std::vector<SubSocket*> socks;

  std::map<SubSocket*, int> qlog_counter;
  std::map<SubSocket*, int> qlog_freqs;

  for (const auto& it : services) {
    std::string name = it.name;
    if (name == "frame" || name == "wideFrame" || (name == "frontFrame" && record_front)) {
      continue;
    }

    if (it.should_log) {
      SubSocket * sock = SubSocket::create(s.ctx, name);
      assert(sock != NULL);
      poller->registerSocket(sock);
      socks.push_back(sock);

      qlog_counter[sock] = (it.decimation == -1) ? -1 : 0;
      qlog_freqs[sock] = it.decimation;
    }
  }

  {
    auto words = gen_init_data();
    auto bytes = words.asBytes();
    logger_init(&s.logger, "rlog", bytes.begin(), bytes.size(), true);
    std::string segment_path;
    int err = logger_next(&s.logger, LOG_ROOT, &segment_path);
    assert(err == 0);
    LOGW("logging to %s", segment_path.c_str());
  }

  std::vector<std::thread> encoder_threads;
#ifndef DISABLE_ENCODER
  if (is_qcom2) {
    // hard code it for convenience
    s.num_encoder_need_sync = 2 + record_front;  // sync all cameras on tici
    // wide camera
    encoder_threads.push_back(std::thread(encoder_thread, false, LOG_CAMERA_ID_ECAMERA));
  } else {
    s.num_encoder_need_sync = 1;
  }
  // rear camera
  encoder_threads.push_back(std::thread(encoder_thread, false, LOG_CAMERA_ID_FCAMERA));
  // front camera
  if (record_front) {
    encoder_threads.push_back(std::thread(encoder_thread, false, LOG_CAMERA_ID_DCAMERA));
  }
#endif

  uint64_t msg_count = 0;
  uint64_t bytes_count = 0;
  Message* msg = nullptr;

  double start_ts = seconds_since_boot();
  s.last_camera_seen_tms = s.last_rotate_tms = millis_since_boot();

  while (!do_exit) {
    for (auto sock : poller->poll(100)) {
      int& counter = qlog_counter[sock];
      const int freq = qlog_freqs[sock];

      while ((msg = sock->receive(true))) {
        logger_log(&s.logger, (uint8_t*)msg->getData(), msg->getSize(), counter == 0);
        if (counter != -1) {
          counter = (counter + 1) % freq;
        }
        bytes_count += msg->getSize();
        delete msg;

        ++msg_count;
        if ((msg_count % 1000) == 0) {
          double ts = seconds_since_boot();
          LOGD("%lu messages, %.2f msg/sec, %.2f KB/sec", msg_count, msg_count * 1.0 / (ts - start_ts), bytes_count * 0.001 / (ts - start_ts));
        }
        if (msg_count % 3 == 0) {
          rotate_log_if_needed(&s);
        }
      }
    }
    rotate_log_if_needed(&s);
  }

  LOGW("joining threads");
  { // wakeup waiting encoder threads
    std::unique_lock<std::mutex> lk(s.rotate_lock);
    s.num_rotating = s.num_encoder_rotated = 0;
    s.rotate_cv.notify_all();
  }
  for (auto& t : encoder_threads) {
    t.join();
  }
  LOGW("encoder joined");

  logger_close(&s.logger);

  for (auto s : socks){
    delete s;
  }

  delete poller;
  delete s.ctx;
  return 0;
}
