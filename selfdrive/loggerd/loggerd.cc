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
#include <pthread.h>

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
#include "services.h"

#if !(defined(QCOM) || defined(QCOM2))
// no encoder on PC
#define DISABLE_ENCODER
#endif


#ifndef DISABLE_ENCODER
#include "encoder.h"
#include "raw_logger.h"
#endif

#include "cereal/gen/cpp/log.capnp.h"

#define CAM_IDX_FCAM 0
#define CAM_IDX_DCAM 1
#define CAM_IDX_ECAM 2

#define CAMERA_FPS 20
#define SEGMENT_LENGTH 60

#define MAIN_BITRATE 5000000
#ifndef QCOM2
#define DCAM_BITRATE 2500000
#else
#define DCAM_BITRATE MAIN_BITRATE
#endif

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

  std::mutex lock;
  std::condition_variable cv;
  char segment_path[4096];
  uint32_t last_frame_id;
  uint32_t rotate_last_frame_id;
  int rotate_segment;
  int rotate_seq_id;

  pthread_mutex_t rotate_lock;
  int num_encoder;
  int should_close;
  int finish_close;
};
LoggerdState s;

#ifndef DISABLE_ENCODER
void encoder_thread(bool is_streaming, bool raw_clips, int cam_idx) {
  int err;

  // 0:f, 1:d, 2:e
  if (cam_idx == CAM_IDX_DCAM) {
  // TODO: add this back
  #ifndef QCOM2
    std::vector<char> value = read_db_bytes("RecordFront");
    if (value.size() == 0 || value[0] != '1') return;
    LOGW("recording front camera");
  #endif
    set_thread_name("FrontCameraEncoder");
  } else if (cam_idx == CAM_IDX_FCAM) {
    set_thread_name("RearCameraEncoder");
  } else if (cam_idx == CAM_IDX_ECAM) {
    set_thread_name("WideCameraEncoder");
  } else {
    LOGE("unexpected camera index provided");
    assert(false);
  }

  VisionStream stream;

  bool encoder_inited = false;
  EncoderState encoder;
  EncoderState encoder_alt;
  bool has_encoder_alt = false;

  pthread_mutex_lock(&s.rotate_lock);
  int my_idx = s.num_encoder;
  s.num_encoder += 1;
  pthread_mutex_unlock(&s.rotate_lock);

  int encoder_segment = -1;
  int cnt = 0;

  PubSocket *idx_sock = PubSocket::create(s.ctx, cam_idx == CAM_IDX_DCAM ? "frontEncodeIdx" : (cam_idx == CAM_IDX_ECAM ? "wideEncodeIdx" : "encodeIdx"));
  assert(idx_sock != NULL);

  LoggerHandle *lh = NULL;

  while (!do_exit) {
    VisionStreamBufs buf_info;
    if (cam_idx == CAM_IDX_DCAM) {
      err = visionstream_init(&stream, VISION_STREAM_YUV_FRONT, false, &buf_info);
    } else if (cam_idx == CAM_IDX_FCAM) {
      err = visionstream_init(&stream, VISION_STREAM_YUV, false, &buf_info);
    } else if (cam_idx == CAM_IDX_ECAM) {
      err = visionstream_init(&stream, VISION_STREAM_YUV_WIDE, false, &buf_info);
    }
    if (err != 0) {
      LOGD("visionstream connect fail");
      usleep(100000);
      continue;
    }

    if (!encoder_inited) {
      LOGD("encoder init %dx%d", buf_info.width, buf_info.height);
      encoder_init(&encoder, cam_idx == CAM_IDX_DCAM ? "dcamera.hevc" : (cam_idx == CAM_IDX_ECAM ? "ecamera.hevc" : "fcamera.hevc"), buf_info.width, buf_info.height, CAMERA_FPS, cam_idx == CAM_IDX_DCAM ? DCAM_BITRATE:MAIN_BITRATE, true, false);

      #ifndef QCOM2
      // TODO: fix qcamera on tici
      if (cam_idx == CAM_IDX_FCAM) {
        encoder_init(&encoder_alt, "qcamera.ts", 480, 360, CAMERA_FPS, 128000, false, true);
        has_encoder_alt = true;
      }
      #endif
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

      //uint64_t current_time = nanos_since_boot();
      //uint64_t diff = current_time - extra.timestamp_eof;
      //double msdiff = (double) diff / 1000000.0;
      // printf("logger latency to tsEof: %f\n", msdiff);

      uint8_t *y = (uint8_t*)buf->addr;
      uint8_t *u = y + (buf_info.width*buf_info.height);
      uint8_t *v = u + (buf_info.width/2)*(buf_info.height/2);

      {
        // all the rotation stuff
        bool should_rotate = false;
        std::unique_lock<std::mutex> lk(s.lock);
        if (cam_idx == CAM_IDX_FCAM) { // TODO: should wait for three cameras on tici?
          // wait if log camera is older on back camera
          while ( extra.frame_id > s.last_frame_id //if the log camera is older, wait for it to catch up.
                 && (extra.frame_id-s.last_frame_id) < 8 // but if its too old then there probably was a discontinuity (visiond restarted)
                 && !do_exit) {
            s.cv.wait(lk);
          }
          should_rotate = extra.frame_id > s.rotate_last_frame_id && encoder_segment < s.rotate_segment && s.rotate_seq_id == my_idx;
        } else {
          // front camera is best effort
          should_rotate = encoder_segment < s.rotate_segment && s.rotate_seq_id == my_idx;
        }
        if (do_exit) break;

        // rotate the encoder if the logger is on a newer segment
        if (should_rotate) {
          LOG("rotate encoder to %s", s.segment_path);

          encoder_rotate(&encoder, s.segment_path, s.rotate_segment);
          s.rotate_seq_id = (my_idx + 1) % s.num_encoder;

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

        if (encoder.rotating) {
          pthread_mutex_lock(&s.rotate_lock);
          s.should_close += 1;
          pthread_mutex_unlock(&s.rotate_lock);

          while(s.should_close > 0 && s.should_close < s.num_encoder) {
            // printf("%d waiting for others to reach close, %d/%d \n", my_idx, s.should_close, s.num_encoder);
            s.cv.wait(lk);
          }

          pthread_mutex_lock(&s.rotate_lock);
          if (s.should_close == s.num_encoder) {
            s.should_close = 1 - s.num_encoder;
          } else {
            s.should_close += 1;
          }
          encoder_close(&encoder);
          encoder_open(&encoder, encoder.next_path);
          encoder.segment = encoder.next_segment;
          encoder.rotating = false;
          if (has_encoder_alt) {
            encoder_close(&encoder_alt);
            encoder_open(&encoder_alt, encoder_alt.next_path);
            encoder_alt.segment = encoder_alt.next_segment;
            encoder_alt.rotating = false;
          }
          s.finish_close += 1;
          pthread_mutex_unlock(&s.rotate_lock);

          while(s.finish_close > 0 && s.finish_close < s.num_encoder) {
            // printf("%d waiting for others to actually close, %d/%d \n", my_idx, s.finish_close, s.num_encoder);
            s.cv.wait(lk);
          }
          s.finish_close = 0;
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
        MessageBuilder msg;
        auto eidx = msg.initEvent().initEncodeIdx();
        eidx.setFrameId(extra.frame_id);
#ifdef QCOM2
        eidx.setType(cereal::EncodeIndex::Type::FULL_H_E_V_C);
#else
        eidx.setType(cam_idx == CAM_IDX_DCAM ? cereal::EncodeIndex::Type::FRONT : cereal::EncodeIndex::Type::FULL_H_E_V_C);
#endif
        eidx.setEncodeId(cnt);
        eidx.setSegmentNum(out_segment);
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

  std::vector<char> git_commit = read_db_bytes("GitCommit");
  if (git_commit.size() > 0) {
    init.setGitCommit(capnp::Text::Reader(git_commit.data(), git_commit.size()));
  }

  std::vector<char> git_branch = read_db_bytes("GitBranch");
  if (git_branch.size() > 0) {
    init.setGitBranch(capnp::Text::Reader(git_branch.data(), git_branch.size()));
  }

  std::vector<char> git_remote = read_db_bytes("GitRemote");
  if (git_remote.size() > 0) {
    init.setGitRemote(capnp::Text::Reader(git_remote.data(), git_remote.size()));
  }

  std::vector<char> passive = read_db_bytes("Passive");
  init.setPassive(passive.size() > 0 && passive[0] == '1');
  {
    // log params
    std::map<std::string, std::string> params;
    read_db_all(&params);
    auto lparams = init.initParams().initEntries(params.size());
    int i = 0;
    for (auto& kv : params) {
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

  err = logger_next(&s.logger, LOG_ROOT, s.segment_path, sizeof(s.segment_path), &s.rotate_segment);
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

    auto bytes = msg.toBytes();
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

  int segment_length = SEGMENT_LENGTH;
  if (getenv("LOGGERD_TEST")) {
    segment_length = atoi(getenv("LOGGERD_SEGMENT_LENGTH"));
  }

  setpriority(PRIO_PROCESS, 0, -12);

  clear_locks();

  signal(SIGINT, (sighandler_t)set_do_exit);
  signal(SIGTERM, (sighandler_t)set_do_exit);

  s.ctx = Context::create();
  Poller * poller = Poller::create();

  // subscribe to all services

  SubSocket *frame_sock = NULL;
  std::vector<SubSocket*> socks;

  std::map<SubSocket*, int> qlog_counter;
  std::map<SubSocket*, int> qlog_freqs;

  for (const auto& it : services) {
    std::string name = it.name;

    if (it.should_log) {
      SubSocket * sock = SubSocket::create(s.ctx, name);
      assert(sock != NULL);

      poller->registerSocket(sock);
      socks.push_back(sock);

      if (name == "frame") {
        frame_sock = sock;
      }

      qlog_counter[sock] = (it.decimation == -1) ? -1 : 0;
      qlog_freqs[sock] = it.decimation;
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
  s.rotate_seq_id = 0;
  s.should_close = 0;
  s.finish_close = 0;
  s.num_encoder = 0;
  pthread_mutex_init(&s.rotate_lock, NULL);
#ifndef DISABLE_ENCODER
  // rear camera
  std::thread encoder_thread_handle(encoder_thread, is_streaming, false, CAM_IDX_FCAM);

  // front camera
  std::thread front_encoder_thread_handle(encoder_thread, false, false, CAM_IDX_DCAM);

  #ifdef QCOM2
  // wide camera
  std::thread wide_encoder_thread_handle(encoder_thread, false, false, CAM_IDX_ECAM);
  #endif
#endif

  uint64_t msg_count = 0;
  uint64_t bytes_count = 0;

  while (!do_exit) {
    for (auto sock : poller->poll(100 * 1000)) {
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
    if (ts - last_rotate_ts > segment_length) {
      // rotate the log

      last_rotate_ts += segment_length;

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
  #ifdef QCOM2
  wide_encoder_thread_handle.join();
  #endif
  front_encoder_thread_handle.join();
  encoder_thread_handle.join();
  LOGW("encoder joined");
#endif

  logger_close(&s.logger);

  for (auto s : socks){
    delete s;
  }

  delete poller;
  delete s.ctx;
  return 0;
}
