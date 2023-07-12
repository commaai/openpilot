#include <unordered_map>

#include "system/loggerd/encoder/encoder.h"
#include "system/loggerd/encoder_writer.h"
#include "system/loggerd/loggerd.h"
#include "system/loggerd/video_writer.h"

ExitHandler do_exit;

struct LoggerdState {
  LoggerState logger = {};
  char segment_path[4096];
  double last_camera_seen_tms;
  double last_rotate_tms = 0.;      // last rotate time in ms
  std::unordered_map<std::string, std::unique_ptr<EncoderWriter>> remote_encoders;
};

void logger_rotate(LoggerdState *s) {
  int segment = -1;
  int err = logger_next(&s->logger, LOG_ROOT.c_str(), s->segment_path, sizeof(s->segment_path), &segment);
  assert(err == 0);

  for (auto &[_, encoder] : s->remote_encoders) {
    encoder->rotate(s->segment_path);
  }
  EncoderWriter::ready_to_rotate = 0;

  s->last_rotate_tms = millis_since_boot();
  LOGW((s->logger.part == 0) ? "logging to %s" : "rotated to %s", s->segment_path);
}

void rotate_if_needed(LoggerdState *s) {
  // all encoders ready, trigger rotation
  bool all_ready = s->remote_encoders.size() > 0 && (EncoderWriter::ready_to_rotate == s->remote_encoders.size());

  // fallback logic to prevent extremely long segments in the case of camera, encoder, etc. malfunctions
  bool timed_out = false;
  double tms = millis_since_boot();
  double seg_length_secs = (tms - s->last_rotate_tms) / 1000.;
  if ((seg_length_secs > SEGMENT_LENGTH) && !LOGGERD_TEST) {
    // TODO: might be nice to put these reasons in the sentinel
    if ((tms - s->last_camera_seen_tms) > NO_CAMERA_PATIENCE) {
      timed_out = true;
      LOGE("no camera packets seen. auto rotating");
    } else if (seg_length_secs > SEGMENT_LENGTH*1.2) {
      timed_out = true;
      LOGE("segment too long. auto rotating");
    }
  }

  if (all_ready || timed_out) {
    logger_rotate(s);
  }
}

size_t handle_encoder_msg(LoggerdState *s, const std::string &name, Message *m) {
  auto &encoder = s->remote_encoders[name];
  if (!encoder) {
    for (const auto &cam : cameras_logged) {
      for (const auto &encoder_info : cam.encoder_infos) {
        if (name == encoder_info.publish_name) {
          encoder.reset(new EncoderWriter(s->segment_path, encoder_info));
          break;
        }
      }
    }
    assert(encoder != nullptr);
  }
  return encoder->write(&(s->logger), m);
}

void loggerd_thread() {
  // setup messaging
  typedef struct QlogState {
    std::string name;
    int counter, freq;
    bool encoder;
  } QlogState;
  std::unordered_map<SubSocket*, QlogState> qlog_states;

  std::unique_ptr<Context> ctx(Context::create());
  std::unique_ptr<Poller> poller(Poller::create());

  // subscribe to all socks
  for (const auto& it : services) {
    const bool encoder = strcmp(it.name+strlen(it.name)-strlen("EncodeData"), "EncodeData") == 0;
    if (!it.should_log && !encoder) continue;
    LOGD("logging %s (on port %d)", it.name, it.port);

    SubSocket * sock = SubSocket::create(ctx.get(), it.name);
    assert(sock != NULL);
    poller->registerSocket(sock);
    qlog_states[sock] = {
      .name = it.name,
      .counter = 0,
      .freq = it.decimation,
      .encoder = encoder,
    };
  }

  LoggerdState s;
  // init logger
  logger_init(&s.logger, true);
  logger_rotate(&s);
  Params().put("CurrentRoute", s.logger.route_name);

  uint64_t msg_count = 0, bytes_count = 0;
  double start_ts = millis_since_boot();
  while (!do_exit) {
    // poll for new messages on all sockets
    for (auto sock : poller->poll(1000)) {
      if (do_exit) break;

      // drain socket
      int count = 0;
      QlogState &qs = qlog_states[sock];
      Message *msg = nullptr;
      while (!do_exit && (msg = sock->receive(true))) {
        const bool in_qlog = qs.freq != -1 && (qs.counter++ % qs.freq == 0);

        if (qs.encoder) {
          s.last_camera_seen_tms = millis_since_boot();
          bytes_count += handle_encoder_msg(&s, qs.name, msg);
        } else {
          logger_log(&s.logger, (uint8_t *)msg->getData(), msg->getSize(), in_qlog);
          bytes_count += msg->getSize();
          delete msg;
        }

        rotate_if_needed(&s);

        if ((++msg_count % 1000) == 0) {
          double seconds = (millis_since_boot() - start_ts) / 1000.0;
          LOGD("%lu messages, %.2f msg/sec, %.2f KB/sec", msg_count, msg_count / seconds, bytes_count * 0.001 / seconds);
        }

        count++;
        if (count >= 200) {
          LOGD("large volume of '%s' messages", qs.name.c_str());
          break;
        }
      }
    }
  }

  LOGW("closing logger");
  logger_close(&s.logger, &do_exit);

  if (do_exit.power_failure) {
    LOGE("power failure");
    sync();
    LOGE("sync done");
  }

  // messaging cleanup
  for (auto &[sock, qs] : qlog_states) delete sock;
}

int main(int argc, char** argv) {
  if (!Hardware::PC()) {
    int ret;
    ret = util::set_core_affinity({0, 1, 2, 3});
    assert(ret == 0);
    // TODO: why does this impact camerad timings?
    //ret = util::set_realtime_priority(1);
    //assert(ret == 0);
  }

  loggerd_thread();

  return 0;
}
