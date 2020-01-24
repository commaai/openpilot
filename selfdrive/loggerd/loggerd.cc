#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <poll.h>
#include <string.h>
#include <inttypes.h>
#include <libyuv.h>
#include <sys/resource.h>

#include <string>
#include <iostream>
#include <fstream>
#include <streambuf>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <random>

#include <ftw.h>

#include <zmq.h>
#include <yaml-cpp/yaml.h>
#include <capnp/serialize.h>

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

#include "logger.h"
#include "messaging.hpp"

#ifndef QCOM
// no encoder on PC
#define DISABLE_ENCODER
#endif


#ifndef DISABLE_ENCODER
#include "encoder.h"
#include "raw_logger.h"
#endif

#include "cereal/gen/cpp/log.capnp.h"

#define CAMERA_FPS 20
#define SEGMENT_LENGTH 60
#define LOG_ROOT "/data/media/0/realdata"
#define ENABLE_LIDAR 0

#define RAW_CLIP_LENGTH 100 // 5 seconds at 20fps
#define RAW_CLIP_FREQUENCY (randrange(61, 8*60)) // once every ~4 minutes

namespace {

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

  std::mutex lock;
  std::condition_variable cv;
  char segment_path[4096];
  uint32_t last_frame_id;
  uint32_t rotate_last_frame_id;
  int rotate_segment;
};
LoggerdState s;

#ifndef DISABLE_ENCODER
void encoder_thread(bool is_streaming, bool raw_clips, bool front) {
  int err;

  if (front) {
    char *value;
    const int result = read_db_value(NULL, "RecordFront", &value, NULL);
    if (result != 0) return;
    if (value[0] != '1') { free(value); return; }
    free(value);
    LOGW("recording front camera");

    set_thread_name("FrontCameraEncoder");
  } else {
    set_thread_name("RearCameraEncoder");
  }

  VisionStream stream;

  bool encoder_inited = false;
  EncoderState encoder;
  EncoderState encoder_alt;
  bool has_encoder_alt = false;

  int encoder_segment = -1;
  int cnt = 0;

  PubSocket *idx_sock = PubSocket::create(s.ctx, front ? "frontEncodeIdx" : "encodeIdx");
  assert(idx_sock != NULL);

  LoggerHandle *lh = NULL;

  while (!do_exit) {
    VisionStreamBufs buf_info;
    if (front) {
      err = visionstream_init(&stream, VISION_STREAM_YUV_FRONT, false, &buf_info);
    } else {
      err = visionstream_init(&stream, VISION_STREAM_YUV, false, &buf_info);
    }
    if (err != 0) {
      LOGD("visionstream connect fail");
      usleep(100000);
      continue;
    }

    if (!encoder_inited) {
      LOGD("encoder init %dx%d", buf_info.width, buf_info.height);
      encoder_init(&encoder, front ? "dcamera.hevc" : "fcamera.hevc", buf_info.width, buf_info.height, CAMERA_FPS, front ? 2500000 : 5000000, true, false);
      if (!front) {
        encoder_init(&encoder_alt, "qcamera.ts", 480, 360, CAMERA_FPS, 128000, false, true);
        has_encoder_alt = true;
      }
      encoder_inited = true;
      if (is_streaming) {
        encoder.zmq_ctx = zmq_ctx_new();
        encoder.stream_sock_raw = zmq_socket(encoder.zmq_ctx, ZMQ_PUB);
        assert(encoder.stream_sock_raw);
        zmq_bind(encoder.stream_sock_raw, "tcp://*:9002");
      }
    }

    // dont log a raw clip in the first minute
    double rawlogger_start_time = seconds_since_boot()+RAW_CLIP_FREQUENCY;
    int rawlogger_clip_cnt = 0;
    RawLogger *rawlogger = NULL;

    if (raw_clips) {
      rawlogger = new RawLogger("prcamera", buf_info.width, buf_info.height, CAMERA_FPS);
    }

    while (!do_exit) {
      VIPCBufExtra extra;
      VIPCBuf* buf = visionstream_get(&stream, &extra);
      if (buf == NULL) {
        LOG("visionstream get failed");
        break;
      }

      uint64_t current_time = nanos_since_boot();
      uint64_t diff = current_time - extra.timestamp_eof;
      double msdiff = (double) diff / 1000000.0;
      // printf("logger latency to tsEof: %f\n", msdiff);

      uint8_t *y = (uint8_t*)buf->addr;
      uint8_t *u = y + (buf_info.width*buf_info.height);
      uint8_t *v = u + (buf_info.width/2)*(buf_info.height/2);

      {
        bool should_rotate = false;
        std::unique_lock<std::mutex> lk(s.lock);
        if (!front) {
          // wait if log camera is older on back camera
          while ( extra.frame_id > s.last_frame_id //if the log camera is older, wait for it to catch up.
                 && (extra.frame_id-s.last_frame_id) < 8 // but if its too old then there probably was a discontinuity (visiond restarted)
                 && !do_exit) {
            s.cv.wait(lk);
          }
          should_rotate = extra.frame_id > s.rotate_last_frame_id && encoder_segment < s.rotate_segment;
        } else {
          // front camera is best effort
          should_rotate = encoder_segment < s.rotate_segment;
        }
        if (do_exit) break;

        // rotate the encoder if the logger is on a newer segment
        if (should_rotate) {
          LOG("rotate encoder to %s", s.segment_path);

          encoder_rotate(&encoder, s.segment_path, s.rotate_segment);
          if (has_encoder_alt) {
            encoder_rotate(&encoder_alt, s.segment_path, s.rotate_segment);
          }

          if (raw_clips) {
            rawlogger->Rotate(s.segment_path, s.rotate_segment);
          }

          encoder_segment = s.rotate_segment;
          if (lh) {
            lh_close(lh);
          }
          lh = logger_get_handle(&s.logger);
        }
      }

      {
        // encode hevc
        int out_segment = -1;
        int out_id = encoder_encode_frame(&encoder,
                                          y, u, v,
                                          buf_info.width, buf_info.height,
                                          &out_segment, &extra);

        if (has_encoder_alt) {
          int out_segment_alt = -1;
          encoder_encode_frame(&encoder_alt,
                               y, u, v,
                               buf_info.width, buf_info.height,
                               &out_segment_alt, &extra);
        }

        // publish encode index
        capnp::MallocMessageBuilder msg;
        cereal::Event::Builder event = msg.initRoot<cereal::Event>();
        event.setLogMonoTime(nanos_since_boot());
        auto eidx = event.initEncodeIdx();
        eidx.setFrameId(extra.frame_id);
        eidx.setType(front ? cereal::EncodeIndex::Type::FRONT : cereal::EncodeIndex::Type::FULL_H_E_V_C);
        eidx.setEncodeId(cnt);
        eidx.setSegmentNum(out_segment);
        eidx.setSegmentId(out_id);

        auto words = capnp::messageToFlatArray(msg);
        auto bytes = words.asBytes();
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
          capnp::MallocMessageBuilder msg;
          cereal::Event::Builder event = msg.initRoot<cereal::Event>();
          event.setLogMonoTime(nanos_since_boot());
          auto eidx = event.initEncodeIdx();
          eidx.setFrameId(extra.frame_id);
          eidx.setType(cereal::EncodeIndex::Type::FULL_LOSSLESS_CLIP);
          eidx.setEncodeId(cnt);
          eidx.setSegmentNum(out_segment);
          eidx.setSegmentId(out_id);

          auto words = capnp::messageToFlatArray(msg);
          auto bytes = words.asBytes();
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
    }

    if (lh) {
      lh_close(lh);
      lh = NULL;
    }

    if (raw_clips) {
      rawlogger->Close();
      delete rawlogger;
    }

    visionstream_destroy(&stream);
  }

  delete idx_sock;

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

#if ENABLE_LIDAR

#include <netinet/in.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#define VELODYNE_DATA_PORT 2368
#define VELODYNE_TELEMETRY_PORT 8308

#define MAX_LIDAR_PACKET 2048

int lidar_thread() {
  // increase kernel max buffer size
  system("sysctl -w net.core.rmem_max=26214400");
  set_thread_name("lidar");

  int sock;
  if ((sock = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    perror("cannot create socket");
    return -1;
  }

  int a = 26214400;
  if (setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &a, sizeof(int)) == -1) {
    perror("cannot set socket opts");
    return -1;
  }

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(struct sockaddr_in));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(VELODYNE_DATA_PORT);
  inet_aton("192.168.5.11", &(addr.sin_addr));

  if (bind(sock, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
    perror("cannot bind LIDAR socket");
    return -1;
  }

  capnp::byte buf[MAX_LIDAR_PACKET];

  while (!do_exit) {
    // receive message
    struct sockaddr from;
    socklen_t fromlen = sizeof(from);
    int cnt = recvfrom(sock, (void *)buf, MAX_LIDAR_PACKET, 0, &from, &fromlen);
    if (cnt <= 0) {
      printf("bug in lidar recieve!\n");
      continue;
    }

    // create message for log
    capnp::MallocMessageBuilder msg;
    auto event = msg.initRoot<cereal::Event>();
    event.setLogMonoTime(nanos_since_boot());
    auto lidar_pts = event.initLidarPts();

    // copy in the buffer
    // TODO: can we remove this copy? does it matter?
    kj::ArrayPtr<capnp::byte> bufferPtr = kj::arrayPtr(buf, cnt);
    lidar_pts.setPkt(bufferPtr);

    // log it
    auto words = capnp::messageToFlatArray(msg);
    auto bytes = words.asBytes();
    logger_log(&s.logger, bytes.begin(), bytes.size());
  }
  return 0;
}
#endif

}

void append_property(const char* key, const char* value, void *cookie) {
  std::vector<std::pair<std::string, std::string> > *properties =
    (std::vector<std::pair<std::string, std::string> > *)cookie;

  properties->push_back(std::make_pair(std::string(key), std::string(value)));
}

kj::Array<capnp::word> gen_init_data() {
  capnp::MallocMessageBuilder msg;
  auto event = msg.initRoot<cereal::Event>();
  event.setLogMonoTime(nanos_since_boot());
  auto init = event.initInitData();

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

  char* git_commit = NULL;
  read_db_value(NULL, "GitCommit", &git_commit, NULL);
  if (git_commit) {
    init.setGitCommit(capnp::Text::Reader(git_commit));
  }

  char* git_branch = NULL;
  read_db_value(NULL, "GitBranch", &git_branch, NULL);
  if (git_branch) {
    init.setGitBranch(capnp::Text::Reader(git_branch));
  }

  char* git_remote = NULL;
  read_db_value(NULL, "GitRemote", &git_remote, NULL);
  if (git_remote) {
    init.setGitRemote(capnp::Text::Reader(git_remote));
  }

  char* passive = NULL;
  read_db_value(NULL, "Passive", &passive, NULL);
  init.setPassive(passive && strlen(passive) && passive[0] == '1');


  {
    // log params
    std::map<std::string, std::string> params;
    read_db_all(NULL, &params);
    auto lparams = init.initParams().initEntries(params.size());
    int i = 0;
    for (auto& kv : params) {
      auto lentry = lparams[i];
      lentry.setKey(kv.first);
      lentry.setValue(kv.second);
      i++;
    }
  }


  auto words = capnp::messageToFlatArray(msg);

  if (git_commit) {
    free((void*)git_commit);
  }

  if (git_branch) {
    free((void*)git_branch);
  }

  if (git_remote) {
    free((void*)git_remote);
  }

  if (passive) {
    free((void*)passive);
  }

  return words;
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

  err = logger_next(&s.logger, LOG_ROOT, s.segment_path, sizeof(s.segment_path), &s.rotate_segment);
  assert(err == 0);
  LOGW("bootlog to %s", s.segment_path);

  {
    capnp::MallocMessageBuilder msg;
    auto event = msg.initRoot<cereal::Event>();
    event.setLogMonoTime(nanos_since_boot());

    auto boot = event.initBoot();

    boot.setWallTimeNanos(nanos_since_epoch());

    std::string lastKmsg = util::read_file("/sys/fs/pstore/console-ramoops");
    boot.setLastKmsg(capnp::Data::Reader((const kj::byte*)lastKmsg.data(), lastKmsg.size()));

    std::string lastPmsg = util::read_file("/sys/fs/pstore/pmsg-ramoops-0");
    boot.setLastPmsg(capnp::Data::Reader((const kj::byte*)lastPmsg.data(), lastPmsg.size()));

    auto words = capnp::messageToFlatArray(msg);
    auto bytes = words.asBytes();
    logger_log(&s.logger, bytes.begin(), bytes.size(), false);
  }

  logger_close(&s.logger);
}

int main(int argc, char** argv) {
  int err;

  if (argc > 1 && strcmp(argv[1], "--bootlog") == 0) {
    bootlog();
    return 0;
  }

  setpriority(PRIO_PROCESS, 0, -12);

  clear_locks();

  signal(SIGINT, (sighandler_t)set_do_exit);
  signal(SIGTERM, (sighandler_t)set_do_exit);

  s.ctx = Context::create();
  Poller * poller = Poller::create();

  std::string exe_dir = util::dir_name(util::readlink("/proc/self/exe"));
  std::string service_list_path = exe_dir + "/../../cereal/service_list.yaml";

  // subscribe to all services

  SubSocket *frame_sock = NULL;
  std::vector<SubSocket*> socks;

  std::map<SubSocket*, int> qlog_counter;
  std::map<SubSocket*, int> qlog_freqs;

  YAML::Node service_list = YAML::LoadFile(service_list_path);
  for (const auto& it : service_list) {
    auto name = it.first.as<std::string>();
    bool should_log = it.second[1].as<bool>();
    int qlog_freq = it.second[3] ? it.second[3].as<int>() : 0;

    if (should_log) {
      SubSocket * sock = SubSocket::create(s.ctx, name);
      assert(sock != NULL);

      poller->registerSocket(sock);
      socks.push_back(sock);

      if (name == "frame") {
        frame_sock = sock;
      }

      qlog_counter[sock] = (qlog_freq == 0) ? -1 : 0;
      qlog_freqs[sock] = qlog_freq;
    }
  }


  {
    auto words = gen_init_data();
    auto bytes = words.asBytes();
    logger_init(&s.logger, "rlog", bytes.begin(), bytes.size(), true);
  }

  bool is_streaming = false;
  bool is_logging = true;

  if (argc > 1 && strcmp(argv[1], "--stream") == 0) {
    is_streaming = true;
  } else if (argc > 1 && strcmp(argv[1], "--only-stream") == 0) {
    is_streaming = true;
    is_logging = false;
  }

  if (is_logging) {
    err = logger_next(&s.logger, LOG_ROOT, s.segment_path, sizeof(s.segment_path), &s.rotate_segment);
    assert(err == 0);
    LOGW("logging to %s", s.segment_path);
  }

  double start_ts = seconds_since_boot();
  double last_rotate_ts = start_ts;

#ifndef DISABLE_ENCODER
  // rear camera
  std::thread encoder_thread_handle(encoder_thread, is_streaming, false, false);

  // front camera
  std::thread front_encoder_thread_handle(encoder_thread, false, false, true);
#endif

#if ENABLE_LIDAR
  std::thread lidar_thread_handle(lidar_thread);
#endif

  uint64_t msg_count = 0;
  uint64_t bytes_count = 0;

  while (!do_exit) {
    for (auto sock : poller->poll(100 * 1000)){
      while (true) {
        Message * msg = sock->receive(true);
        if (msg == NULL){
          break;
        }

        uint8_t* data = (uint8_t*)msg->getData();
        size_t len = msg->getSize();

        if (sock == frame_sock) {
          // track camera frames to sync to encoder
          auto amsg = kj::heapArray<capnp::word>((len / sizeof(capnp::word)) + 1);
          memcpy(amsg.begin(), data, len);

          capnp::FlatArrayMessageReader cmsg(amsg);
          cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();
          if (event.isFrame()) {
            std::unique_lock<std::mutex> lk(s.lock);
            s.last_frame_id = event.getFrame().getFrameId();
            lk.unlock();
            s.cv.notify_all();
          }
        }

        logger_log(&s.logger, data, len, qlog_counter[sock] == 0);
        delete msg;

        if (qlog_counter[sock] != -1) {
          //printf("%p: %d/%d\n", socks[i], qlog_counter[socks[i]], qlog_freqs[socks[i]]);
          qlog_counter[sock]++;
          qlog_counter[sock] %= qlog_freqs[sock];
        }

        bytes_count += len;
        msg_count++;
      }
    }

    double ts = seconds_since_boot();
    if (ts - last_rotate_ts > SEGMENT_LENGTH) {
      // rotate the log

      last_rotate_ts += SEGMENT_LENGTH;

      std::lock_guard<std::mutex> guard(s.lock);
      s.rotate_last_frame_id = s.last_frame_id;

      if (is_logging) {
        err = logger_next(&s.logger, LOG_ROOT, s.segment_path, sizeof(s.segment_path), &s.rotate_segment);
        assert(err == 0);
        LOGW("rotated to %s", s.segment_path);
      }
    }

    if ((msg_count%1000) == 0) {
      LOGD("%lu messages, %.2f msg/sec, %.2f KB/sec", msg_count, msg_count*1.0/(ts-start_ts), bytes_count*0.001/(ts-start_ts));
    }
  }

  LOGW("joining threads");
  s.cv.notify_all();


#ifndef DISABLE_ENCODER
  front_encoder_thread_handle.join();
  encoder_thread_handle.join();
  LOGW("encoder joined");
#endif

#if ENABLE_LIDAR
  lidar_thread_handle.join();
  LOGW("lidar joined");
#endif

  logger_close(&s.logger);

  for (auto s : socks){
    delete s;
  }

  delete s.ctx;
  return 0;
}
