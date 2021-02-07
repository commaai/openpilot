#include <cassert>
#include <sys/resource.h>

#include <ftw.h>
#include <algorithm>
#include "common/timing.h"
#include "common/params.h"
#include "common/swaglog.h"
#include "common/util.h"
#include "camerad/cameras/camera_common.h"
#include "logger.h"
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
constexpr int MAIN_BITRATE = 5000000;
constexpr int DCAM_BITRATE = 2500000;
const bool IS_QCOM2 = false;
constexpr int DCAM_BITRATE = 2500000;
#else
constexpr int MAIN_BITRATE = 10000000;
constexpr int DCAM_BITRATE = MAIN_BITRATE;
const bool IS_QCOM2 = true;
#endif

#define NO_CAMERA_PATIENCE 500 // fall back to time-based rotation if all cameras are dead

const int SEGMENT_LENGTH = getenv("LOGGERD_TEST") ? atoi(getenv("LOGGERD_SEGMENT_LENGTH")) : 60;

ExitHandler do_exit;

LogCameraInfo cameras_logged[] = {
  {.id = F_CAMERA,
    .stream_type = VISION_STREAM_YUV_BACK,
    .filename = "fcamera.hevc",
    .frame_packet_name = "roadCameraState",
    .fps = MAIN_FPS,
    .bitrate = MAIN_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = true},
  {.id = D_CAMERA,
    .stream_type = VISION_STREAM_YUV_FRONT,
    .filename = "dcamera.hevc",
    .frame_packet_name = "driverCameraState",
    .fps = MAIN_FPS, // on EONs, more compressed this way
    .bitrate = DCAM_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = false},
#ifdef QCOM2
  {.id = E_CAMERA,
    .stream_type = VISION_STREAM_YUV_WIDE,
    .filename = "ecamera.hevc",
    .frame_packet_name = "wideRoadCameraState",
    .fps = MAIN_FPS,
    .bitrate = MAIN_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = false},
#endif
};
const LogCameraInfo qcam_info = {
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


typedef struct QlogState {
  int counter, freq;
} QlogState;

class EncoderState {
public:
  EncoderState(const LogCameraInfo &ci, SubSocket *sock, const QlogState &qs, bool need_waiting)
      : ci(ci), frame_sock(sock), qlog_state(qs), need_waiting(need_waiting) {
    last_camera_seen_tms = millis_since_boot();
    thread = std::thread(&EncoderState::encoder_thread, this);
  }
  ~EncoderState() { thread.join(); }
  void encoder_thread();
  LogCameraInfo ci;
  std::unique_ptr<SubSocket> frame_sock;
  QlogState qlog_state;
  std::atomic<double> last_camera_seen_tms;
  std::thread thread;
  const bool need_waiting;
};

struct LoggerdState {
  Context *ctx;
  LoggerState logger = {};
  char segment_path[4096];
  int encoders_max_waiting = 0;
  int rotate_segment;
  double last_rotate_tms;
  std::mutex rotate_lock;
  std::atomic<int> encoders_waiting;
  std::condition_variable cv;
  std::vector<EncoderState *> encoder_states;
};
LoggerdState s;

void drain_socket(LoggerHandle *lh, SubSocket *sock, QlogState &qs) {
  if (!lh) return;

  Message *msg = nullptr;
  while (!do_exit && (msg = sock->receive(true))) {
    lh_log(lh, (uint8_t *)msg->getData(), msg->getSize(), qs.counter == 0 && qs.freq != -1);
    if (qs.freq != -1) {
      qs.counter = (qs.counter + 1) % qs.freq;
    }
    delete msg;
  }
}

void EncoderState::encoder_thread() {
  set_thread_name(ci.filename);

  uint32_t total_frame_cnt = 0;
  LoggerHandle *lh = NULL;
  std::vector<Encoder *> encoders;
  int encoder_segment = -1;
  std::string segment_path;
  const int max_segment_frames = SEGMENT_LENGTH * MAIN_FPS;
 
  VisionIpcClient vipc_client = VisionIpcClient("camerad", ci.stream_type, false);
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
      encoders.push_back(new Encoder(ci.filename, buf_info.width, buf_info.height,
                                     ci.fps, ci.bitrate, ci.is_h265, ci.downscale));
      // qcamera encoder
      if (ci.has_qcamera) {
        encoders.push_back(new Encoder(qcam_info.filename, qcam_info.frame_width, qcam_info.frame_height,
                                       qcam_info.fps, qcam_info.bitrate, qcam_info.is_h265, qcam_info.downscale));
      }
    }

    while (!do_exit) {
      VisionIpcBufExtra extra;
      VisionBuf* buf = vipc_client.recv(&extra);
      if (buf == nullptr) continue;

      bool should_rotate = false;
      {
        std::unique_lock lk(s.rotate_lock);
        last_camera_seen_tms = millis_since_boot();
        // rotate the encoder if the logger is on a newer segment
        should_rotate = (encoder_segment != s.rotate_segment);
        if (!should_rotate && need_waiting && ((total_frame_cnt % max_segment_frames) == 0)) {
          // max_segment_frames have been recorded, need to rotate
          should_rotate = true;
          s.encoders_waiting++;
          s.cv.wait(lk, [&] { return s.encoders_waiting == 0 || do_exit; });
          if (do_exit) break;
        }
        if (should_rotate) {
          encoder_segment = s.rotate_segment;
          segment_path = s.segment_path;
        }
      }
      if (should_rotate) {
        LOGW("camera %d rotate encoder to %s", ci.id, segment_path.c_str());
        if (lh) {
          lh_close(lh);
        }
        lh = logger_get_handle(&s.logger);
        for (auto &e : encoders) {
          e->encoder_close();
          e->encoder_open(segment_path.c_str());
        }
      }

      // log frame socket
      drain_socket(lh, frame_sock.get(), qlog_state);

      // encode a frame
      for (int i = 0; i < encoders.size(); ++i) {
        int out_segment = -1;
        int out_id = encoders[i]->encode_frame(buf->y, buf->u, buf->v,
                                               buf->width, buf->height,
                                               &out_segment, extra.timestamp_eof);
        if (i == 0 && out_id != -1) {
          // publish encode index
          MessageBuilder msg;
          // this is really ugly
          auto eidx = ci.id == D_CAMERA ? msg.initEvent().initFrontEncodeIdx() : (ci.id == E_CAMERA ? msg.initEvent().initWideEncodeIdx() : msg.initEvent().initEncodeIdx());
          eidx.setFrameId(extra.frame_id);
          eidx.setTimestampSof(extra.timestamp_sof);
          eidx.setTimestampEof(extra.timestamp_eof);
          eidx.setType((IS_QCOM2 || ci.id != D_CAMERA) ? cereal::EncodeIndex::Type::FULL_H_E_V_C : cereal::EncodeIndex::Type::FRONT);
          eidx.setEncodeId(total_frame_cnt);
          eidx.setSegmentNum(out_segment);
          eidx.setSegmentId(out_id);
          if (lh) {
            auto bytes = msg.toBytes();
            lh_log(lh, bytes.begin(), bytes.size(), false);
          }
        }
      }
      ++total_frame_cnt;
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

void loggerd_logger_next() {
  s.last_rotate_tms = millis_since_boot();
  assert(0 == logger_next(&s.logger, LOG_ROOT.c_str(), s.segment_path, sizeof(s.segment_path), &s.rotate_segment));
  LOGW((s.rotate_segment == 0) ? "logging to %s" : "rotated to %s", s.segment_path);
}

bool loggerd_should_rotate() {
  bool should_rotate = s.encoders_waiting >= s.encoders_max_waiting;
  if (!should_rotate) {
    const double tms = millis_since_boot();
    if ((tms - s.last_rotate_tms) >= (SEGMENT_LENGTH * 1000)) {
      auto timeout_encoder = std::find_if(s.encoder_states.begin(), s.encoder_states.end(), [&](EncoderState *es) {
        return es->need_waiting && (tms - es->last_camera_seen_tms) >= NO_CAMERA_PATIENCE;
      });
      if (timeout_encoder != s.encoder_states.end()) {
        should_rotate = true;
        LOGW("no camera %d packet seen. auto rotated", (*timeout_encoder)->ci.id);
      }
    }
  }
  return should_rotate && !do_exit;
}

} // namespace

int main(int argc, char** argv) {
  setpriority(PRIO_PROCESS, 0, -12);

  clear_locks();

  // init and open logger
  logger_init(&s.logger, "rlog", true);
  loggerd_logger_next();

  // setup messaging
  std::map<SubSocket*, QlogState> qlog_states;
  s.ctx = Context::create();
  Poller * poller = Poller::create();
  const bool record_front = Params().read_db_bool("RecordFront");
  // subscribe to all socks
  for (const auto& it : services) {
    if (!it.should_log) continue;

    SubSocket * sock = SubSocket::create(s.ctx, it.name);
    assert(sock != NULL);
    QlogState qs = {.counter = 0, .freq = it.decimation};

    auto camra_info = std::find_if(std::begin(cameras_logged), std::end(cameras_logged), [&](LogCameraInfo &ci) {
      return strcmp(it.name, ci.frame_packet_name) == 0 && (ci.id != D_CAMERA || record_front);
    });
    if (camra_info != std::end(cameras_logged)) {
      // init and start encoder thread
      const bool need_waiting = (IS_QCOM2 || camra_info->id != D_CAMERA);
      s.encoders_max_waiting += need_waiting;
      s.encoder_states.push_back(new EncoderState(*camra_info, sock, qs, need_waiting));
    } else {
      poller->registerSocket(sock);
      qlog_states[sock] = qs;
    }
  }

  while (!do_exit) {
    for (auto sock : poller->poll(1000)) {
      drain_socket(s.logger.cur_handle, sock, qlog_states[sock]);
    }

    if (loggerd_should_rotate()) {
      { // rotate to new segment
        std::unique_lock lk(s.rotate_lock);
        loggerd_logger_next();
        s.encoders_waiting = 0;
      }
      s.cv.notify_all();
    }
  }

  LOGW("closing encoders");
  s.encoders_waiting = 0;
  s.cv.notify_all();
  for (auto &[sock, qs] : qlog_states) delete sock;
  for (auto &e : s.encoder_states) { delete e; }

  LOGW("closing logger");
  logger_close(&s.logger);

  delete poller;
  delete s.ctx;

  return 0;
}
