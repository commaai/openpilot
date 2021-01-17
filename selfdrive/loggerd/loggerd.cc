#include "loggerd.h"
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

static bool file_exists(const std::string& fn) {
  std::ifstream f(fn);
  return f.good();
}

void encoder_thread(LoggerdState *s, int cam_idx) {
  assert(cam_idx < LOG_CAMERA_ID_MAX-1);

  LogCameraInfo &cam_info = cameras_logged[cam_idx];
  RotateState &rotate_state = s->rotate_state[cam_idx];

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
        s->rotate_lock.lock();
        s->rotate_lock.unlock();

        // wait if camera pkt id is older than stream
        rotate_state.waitLogThread();

        if (do_exit) break;

        // rotate the encoder if the logger is on a newer segment
        if (rotate_state.should_rotate) {
          LOGW("camera %d rotate encoder to %s", cam_idx, s->segment_path);

          if (!rotate_state.initialized) {
            rotate_state.last_rotate_frame_id = extra.frame_id - 1;
            rotate_state.initialized = true;
          }

          if (lh) {
            lh_close(lh);
          }
          lh = logger_get_handle(&s->logger);

          {
            std::unique_lock lk(s->rotate_lock);
            for (auto &e : encoders) {
              e->encoder_close();
              e->encoder_open(s->segment_path, s->rotate_segment);
            }
          }
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
  LoggerState logger;
  {
    auto words = gen_init_data();
    auto bytes = words.asBytes();
    logger_init(&logger, "bootlog", bytes.begin(), bytes.size(), false);
  }
  char segment_path[4096];
  err = logger_next(&logger, LOG_ROOT.c_str(), segment_path, sizeof(segment_path), nullptr);
  assert(err == 0);
  LOGW("bootlog to %s", segment_path);

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
    logger_log(&logger, bytes.begin(), bytes.size(), false);
  }

  logger_close(&logger);
}

LoggerdState::LoggerdState() {
  segment_length = SEGMENT_LENGTH;
  if (getenv("LOGGERD_TEST")) {
    segment_length = atoi(getenv("LOGGERD_SEGMENT_LENGTH"));
  }

  clear_locks();

  // setup messaging
  ctx = Context::create();
  poller = Poller::create();
  // subscribe to all socks
  for (const auto &it : services) {
    if (!it.should_log) continue;

    SubSocket *sock = SubSocket::create(ctx, it.name);
    assert(sock != NULL);
    poller->registerSocket(sock);

    int fpkt_id = -1;
    for (int cid = 0; cid <= MAX_CAM_IDX; cid++) {
      if (std::string(it.name) == cameras_logged[cid].frame_packet_name) {
        fpkt_id = cid;
        break;
      }
    }
    socket_states[sock] = {.counter = 0, .freq = it.decimation, .fpkt_id = fpkt_id};
  }

  // init logger
  {
    auto words = gen_init_data();
    auto bytes = words.asBytes();
    logger_init(&logger, "rlog", bytes.begin(), bytes.size(), true);
  }

  // init encoders
  // TODO: create these threads dynamically on frame packet presence
  encoder_threads.push_back(std::thread(encoder_thread, this, LOG_CAMERA_ID_FCAMERA));
  rotate_state[LOG_CAMERA_ID_FCAMERA].enabled = true;

#if defined(QCOM) || defined(QCOM2)
  bool record_front = Params().read_db_bool("RecordFront");
  if (record_front) {
    encoder_threads.push_back(std::thread(encoder_thread, this, LOG_CAMERA_ID_DCAMERA));
    rotate_state[LOG_CAMERA_ID_DCAMERA].enabled = true;
  }

#ifdef QCOM2
  encoder_threads.push_back(std::thread(encoder_thread, this, LOG_CAMERA_ID_ECAMERA));
  rotate_state[LOG_CAMERA_ID_ECAMERA].enabled = true;
#endif
#endif
}

LoggerdState::~LoggerdState() {
  LOGW("closing encoders");
  for (auto &r : rotate_state) r.cancelWait();
  for (auto &t : encoder_threads) t.join();

  LOGW("closing logger");
  logger_close(&logger);

  // messaging cleanup
  for (auto &v : socket_states) delete v.first;
  delete poller;
  delete ctx;
}

static int get_frame_id(Message *m, int fpkt_id) {
  static kj::Array<capnp::word> buf = kj::heapArray<capnp::word>(1024);
  // track camera frames to sync to encoder
  // only process last frame
  if (const size_t buf_size = m->getSize() / sizeof(capnp::word) + 1;
      buf.size() < buf_size) {
    buf = kj::heapArray<capnp::word>(buf_size);
  }
  memcpy(buf.begin(), (uint8_t *)m->getData(), m->getSize());

  capnp::FlatArrayMessageReader cmsg(buf);
  cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();
  if (fpkt_id == LOG_CAMERA_ID_FCAMERA) {
    return event.getFrame().getFrameId();
  } else if (fpkt_id == LOG_CAMERA_ID_DCAMERA) {
    return event.getFrontFrame().getFrameId();
  } else if (fpkt_id == LOG_CAMERA_ID_ECAMERA) {
    return event.getWideFrame().getFrameId();
  }
  assert(0);
}

void LoggerdState::rotate() {
  bool new_segment = logger.part == -1;
  if (logger.part > -1) {
    double tms = millis_since_boot();
    if (tms - last_camera_seen_tms <= NO_CAMERA_PATIENCE && encoder_threads.size() > 0) {
      new_segment = true;
      for (auto &r : rotate_state) {
        // this *should* be redundant on tici since all camera frames are synced
        new_segment &= (((r.stream_frame_id >= r.last_rotate_frame_id + segment_length * MAIN_FPS) &&
                         (!r.should_rotate) && (r.initialized)) ||
                        (!r.enabled));
#ifndef QCOM2
        break;  // only look at fcamera frame id if not QCOM2
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
    std::unique_lock lk(rotate_lock);
    last_rotate_tms = millis_since_boot();

    int err = logger_next(&logger, LOG_ROOT.c_str(), segment_path, sizeof(segment_path), &rotate_segment);
    assert(err == 0);
    LOGW((logger.part == 0) ? "logging to %s" : "rotated to %s", segment_path);

    // rotate encoders
    for (auto &r : rotate_state) r.rotate();
  }
}

std::unique_ptr<Message> LoggerdState::log(SubSocket *sock, SocketState& ss) {
  static uint64_t msg_count = 0, bytes_count = 0;
  double start_ts = seconds_since_boot();
  std::unique_ptr<Message> last_msg;
  while (!do_exit) {
    Message *msg = sock->receive(true);
    if (!msg) break;

    last_msg.reset(msg);

    logger_log(&logger, (uint8_t *)msg->getData(), msg->getSize(), ss.counter == 0 && ss.freq != -1);
    if (ss.freq != -1) {
      ss.counter = (ss.counter + 1) % ss.freq;
    }

    bytes_count += msg->getSize();
    if ((++msg_count % 1000) == 0) {
      double ts = seconds_since_boot();
      LOGD("%lu messages, %.2f msg/sec, %.2f KB/sec", msg_count, msg_count * 1.0 / (ts - start_ts), bytes_count * 0.001 / (ts - start_ts));
    }
  }
  return last_msg;
}

void LoggerdState::run() {
  last_rotate_tms = last_camera_seen_tms = millis_since_boot();

  while (!do_exit) {
    // TODO: fix msgs from the first poll getting dropped
    // poll for new messages on all sockets
    for (auto sock : poller->poll(1000)) {
      SocketState& ss = socket_states[sock];
      std::unique_ptr<Message> last_msg = log(sock, ss);
      if (last_msg && ss.fpkt_id != -1) {
        // track camera frames to sync to encoder
        // only process last frame
        rotate_state[ss.fpkt_id].setLogFrameId(get_frame_id(last_msg.get(), ss.fpkt_id));
        last_camera_seen_tms = millis_since_boot();
      }
    }
    rotate();
  }
}

int main(int argc, char** argv) {

  setpriority(PRIO_PROCESS, 0, -12);

  if (argc > 1 && strcmp(argv[1], "--bootlog") == 0) {
    bootlog();
    return 0;
  }

  LoggerdState loggerd;
  loggerd.run();

  return 0;
}
