#include <unordered_map>

#include "system/loggerd/encoder/encoder.h"
#include "system/loggerd/loggerd.h"
#include "system/loggerd/video_writer.h"

ExitHandler do_exit;

class LoggerdState;

class RemoteEncoder {
public:
  RemoteEncoder(const EncoderInfo &info) : info(info) {}
  void rotate();
  size_t handle_msg(LoggerdState *s, Message *msg);
  inline static int ready_to_rotate = 0;  // count of encoders ready to rotate

private:
  int write_encode_data(LoggerdState *s, const cereal::Event::Reader event);

  std::unique_ptr<VideoWriter> writer;
  int remote_encoder_segment = -1;
  int current_encoder_segment = -1;
  std::vector<Message *> q;
  int dropped_frames = 0;
  bool recording = false;
  bool marked_ready_to_rotate = false;
  EncoderInfo info;
};

struct LoggerdState {
  LoggerState logger = {};
  char segment_path[4096];
  double last_camera_seen_tms;
  double last_rotate_tms = 0.;      // last rotate time in ms
  std::unordered_map<std::string, std::unique_ptr<RemoteEncoder>> remote_encoders;
};

void logger_rotate(LoggerdState *s) {
  int segment = -1;
  int err = logger_next(&s->logger, LOG_ROOT.c_str(), s->segment_path, sizeof(s->segment_path), &segment);
  assert(err == 0);

  for (auto &[_, encoder] : s->remote_encoders) {
    encoder->rotate();
  }
  RemoteEncoder::ready_to_rotate = 0;

  s->last_rotate_tms = millis_since_boot();
  LOGW((s->logger.part == 0) ? "logging to %s" : "rotated to %s", s->segment_path);
}

void rotate_if_needed(LoggerdState *s) {
  // all encoders ready, trigger rotation
  bool all_ready = s->remote_encoders.size() > 0 && (RemoteEncoder::ready_to_rotate == s->remote_encoders.size());

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

void RemoteEncoder::rotate() {
  writer.reset();
  recording = false;
  current_encoder_segment = remote_encoder_segment;
  marked_ready_to_rotate = false;
}

size_t RemoteEncoder::handle_msg(LoggerdState *s, Message *msg) {
  capnp::FlatArrayMessageReader cmsg(kj::ArrayPtr<capnp::word>((capnp::word *)msg->getData(), msg->getSize() / sizeof(capnp::word)));
  auto event = cmsg.getRoot<cereal::Event>();
  auto idx = (event.*(info.get_encode_data_func))().getIdx();

  remote_encoder_segment = idx.getSegmentNum();
  if (current_encoder_segment == -1) {
    current_encoder_segment = remote_encoder_segment;
    LOGD("%s: has encoderd offset %d", info.publish_name, current_encoder_segment);
  }

  size_t written = 0;
  if (current_encoder_segment == remote_encoder_segment) {
    if (!q.empty()) {
      for (auto &qmsg : q) {
        capnp::FlatArrayMessageReader msg_reader({(capnp::word *)qmsg->getData(), qmsg->getSize() / sizeof(capnp::word)});
        written += write_encode_data(s, msg_reader.getRoot<cereal::Event>());
        delete qmsg;
      }
      q.clear();
    }
    written = write_encode_data(s, event);
    delete msg;
  } else {
    // rotate to the next segment to sync with remote encoders.
    if (!marked_ready_to_rotate) {
      marked_ready_to_rotate = true;
      ++ready_to_rotate;
    }
    q.push_back(msg);
  }
  return written;
}

int RemoteEncoder::write_encode_data(LoggerdState *s, const cereal::Event::Reader event) {
  auto edata = (event.*(info.get_encode_data_func))();
  const auto idx = edata.getIdx();
  const bool is_key_frame = idx.getFlags() & V4L2_BUF_FLAG_KEYFRAME;

  // if we aren't recording yet, try to start, since we are in the correct segment
  if (!recording) {
    if (is_key_frame) {
      // only create on iframe
      if (dropped_frames) {
        // this should only happen for the first segment, maybe
        LOGW("%s: dropped %d non iframe packets before init", info.publish_name, dropped_frames);
        dropped_frames = 0;
      }
      // if we aren't actually recording, don't create the writer
      if (info.record) {
        writer.reset(new VideoWriter(s->segment_path, info.filename, idx.getType() != cereal::EncodeIndex::Type::FULL_H_E_V_C,
                                     info.frame_width, info.frame_height, info.fps, idx.getType()));
        // write the header
        auto header = edata.getHeader();
        writer->write((uint8_t *)header.begin(), header.size(), idx.getTimestampEof() / 1000, true, false);
      }
      recording = true;
    } else {
      // this is a sad case when we aren't recording, but don't have an iframe
      // nothing we can do but drop the frame
      ++dropped_frames;
      return 0;
    }
  }

  // we have to be recording if we are here
  assert(recording);

  // if we are actually writing the video file, do so
  if (writer) {
    auto data = edata.getData();
    writer->write((uint8_t *)data.begin(), data.size(), idx.getTimestampEof() / 1000, false, is_key_frame);
  }

  // put it in log stream as the idx packet
  MessageBuilder msg;
  auto evt = msg.initEvent(event.getValid());
  evt.setLogMonoTime(event.getLogMonoTime());
  (evt.*(info.set_encode_idx_func))(idx);
  auto bytes = msg.toBytes();
  logger_log(&s->logger, (uint8_t *)bytes.begin(), bytes.size(), true);  // always in qlog?
  return bytes.size();
}

size_t handle_encoder_msg(LoggerdState *s, const std::string &name, Message *m) {
  auto &encoder = s->remote_encoders[name];
  if (!encoder) {
    for (const auto &cam : cameras_logged) {
      for (const auto &encoder_info : cam.encoder_infos) {
        if (name == encoder_info.publish_name) {
          encoder.reset(new RemoteEncoder(encoder_info));
          break;
        }
      }
    }
    assert(encoder != nullptr);
  }
  return encoder->handle_msg(s, m);
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
