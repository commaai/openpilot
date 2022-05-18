#include "selfdrive/loggerd/loggerd.h"
#include "selfdrive/loggerd/video_writer.h"

ExitHandler do_exit;

struct LoggerdState {
  LoggerState logger = {};
  char segment_path[4096];
  std::mutex rotate_lock;
  std::atomic<int> rotate_segment;
  std::atomic<double> last_camera_seen_tms;
  std::atomic<int> ready_to_rotate;  // count of encoders ready to rotate
  int max_waiting = 0;
  double last_rotate_tms = 0.;      // last rotate time in ms
};

void logger_rotate(LoggerdState *s) {
  {
    std::unique_lock lk(s->rotate_lock);
    int segment = -1;
    int err = logger_next(&s->logger, LOG_ROOT.c_str(), s->segment_path, sizeof(s->segment_path), &segment);
    assert(err == 0);
    s->rotate_segment = segment;
    s->ready_to_rotate = 0;
    s->last_rotate_tms = millis_since_boot();
  }
  LOGW((s->logger.part == 0) ? "logging to %s" : "rotated to %s", s->segment_path);
}

void rotate_if_needed(LoggerdState *s) {
  if (s->ready_to_rotate == s->max_waiting) {
    logger_rotate(s);
  }

  double tms = millis_since_boot();
  if ((tms - s->last_rotate_tms) > SEGMENT_LENGTH * 1000 &&
      (tms - s->last_camera_seen_tms) > NO_CAMERA_PATIENCE &&
      !LOGGERD_TEST) {
    LOGW("no camera packet seen. auto rotating");
    logger_rotate(s);
  }
}

struct RemoteEncoder {
  std::unique_ptr<VideoWriter> writer;
  int segment = -1;
  std::vector<Message *> q;
  int logger_segment = -1;
  int dropped_frames = 0;
  bool recording = false;
};

int handle_encoder_msg(LoggerdState *s, Message *msg, std::string &name, struct RemoteEncoder &re) {
  const LogCameraInfo &cam_info = (name == "driverEncodeData") ? cameras_logged[1] :
    ((name == "wideRoadEncodeData") ? cameras_logged[2] :
    ((name == "qRoadEncodeData") ? qcam_info : cameras_logged[0]));

  // rotation happened, process the queue (happens before the current message)
  int bytes_count = 0;
  if (re.logger_segment != s->rotate_segment) {
    re.logger_segment = s->rotate_segment;
    for (auto &qmsg: re.q) {
      bytes_count += handle_encoder_msg(s, qmsg, name, re);
    }
    re.q.clear();
  }

  // extract the message
  capnp::FlatArrayMessageReader cmsg(kj::ArrayPtr<capnp::word>((capnp::word *)msg->getData(), msg->getSize()));
  auto event = cmsg.getRoot<cereal::Event>();
  auto edata = (name == "driverEncodeData") ? event.getDriverEncodeData() :
    ((name == "wideRoadEncodeData") ? event.getWideRoadEncodeData() :
    ((name == "qRoadEncodeData") ? event.getQRoadEncodeData() : event.getRoadEncodeData()));
  auto idx = edata.getIdx();
  auto flags = idx.getFlags();

  if (!re.recording) {
    // only create on iframe
    if (flags & V4L2_BUF_FLAG_KEYFRAME) {
      if (re.dropped_frames) {
        // this should only happen for the first segment, maybe
        LOGD("%s: dropped %d non iframe packets before init", name.c_str(), re.dropped_frames);
        re.dropped_frames = 0;
      }
      // if we aren't recording, don't create the writer
      if (cam_info.record) {
        re.writer.reset(new VideoWriter(s->segment_path,
          cam_info.filename, idx.getType() != cereal::EncodeIndex::Type::FULL_H_E_V_C,
          cam_info.frame_width, cam_info.frame_height, cam_info.fps, idx.getType()));
        // write the header
        auto header = edata.getHeader();
        re.writer->write((uint8_t *)header.begin(), header.size(), idx.getTimestampEof()/1000, true, false);
      }
      re.segment = idx.getSegmentNum();
      re.recording = true;
    } else {
      ++re.dropped_frames;
      return bytes_count;
    }
  }

  if (re.segment != idx.getSegmentNum()) {
    if (re.recording) {
      // encoder is on the next segment, this segment is over so we close the videowriter
      re.writer.reset();
      re.recording = false;
      ++s->ready_to_rotate;
      LOGD("rotate %d -> %d ready %d/%d for %s", re.segment, idx.getSegmentNum(), s->ready_to_rotate.load(), s->max_waiting, name.c_str());
    }
    // queue up all the new segment messages, they go in after the rotate
    re.q.push_back(msg);
  } else {
    if (re.writer) {
      auto data = edata.getData();
      re.writer->write((uint8_t *)data.begin(), data.size(), idx.getTimestampEof()/1000, false, flags & V4L2_BUF_FLAG_KEYFRAME);
    }

    // put it in log stream as the idx packet
    MessageBuilder bmsg;
    auto evt = bmsg.initEvent(event.getValid());
    evt.setLogMonoTime(event.getLogMonoTime());
    if (name == "driverEncodeData") { evt.setDriverEncodeIdx(idx); }
    if (name == "wideRoadEncodeData") { evt.setWideRoadEncodeIdx(idx); }
    if (name == "qRoadEncodeData") { evt.setQRoadEncodeIdx(idx); }
    if (name == "roadEncodeData") { evt.setRoadEncodeIdx(idx); }
    auto new_msg = bmsg.toBytes();
    logger_log(&s->logger, (uint8_t *)new_msg.begin(), new_msg.size(), true);   // always in qlog?
    bytes_count += new_msg.size();

    // this frees the message
    delete msg;
  }

  return bytes_count;
}

void loggerd_thread() {
  // setup messaging
  typedef struct QlogState {
    std::string name;
    int counter, freq;
    bool encoder;
  } QlogState;
  std::unordered_map<SubSocket*, QlogState> qlog_states;
  std::unordered_map<SubSocket*, struct RemoteEncoder> remote_encoders;

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

  // init encoders
  s.last_camera_seen_tms = millis_since_boot();
  for (const auto &cam : cameras_logged) {
    if (cam.enable) {
      s.max_waiting++;
      if (cam.has_qcamera) { s.max_waiting++; }
    }
  }

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
          bytes_count += handle_encoder_msg(&s, msg, qs.name, remote_encoders[sock]);
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
  if (Hardware::TICI()) {
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