#include "selfdrive/loggerd/loggerd.h"
#include "selfdrive/loggerd/video_writer.h"

ExitHandler do_exit;

class RemoteEncoder;

struct LoggerdState {
  LoggerState logger = {};
  char segment_path[4096];
  std::atomic<int> rotate_segment;
  std::atomic<double> last_camera_seen_tms;
  std::atomic<int> ready_to_rotate;  // count of encoders ready to rotate
  int max_waiting = 0;
  double last_rotate_tms = 0.;      // last rotate time in ms
  std::unordered_map<std::string, std::unique_ptr<RemoteEncoder>> encoders;
};

void logger_rotate(LoggerdState *s) {
  int segment = -1;
  int err = logger_next(&s->logger, LOG_ROOT.c_str(), s->segment_path, sizeof(s->segment_path), &segment);
  assert(err == 0);
  s->rotate_segment = segment;
  s->ready_to_rotate = 0;
  s->last_rotate_tms = millis_since_boot();
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

class RemoteEncoder {
 public:
  RemoteEncoder(LoggerdState *s, const std::string &name);
  ~RemoteEncoder() {};
  uint32_t handlePacket(Message *raw_msg);

 protected:
  uint32_t writePacket(cereal::Event::Reader event);
  uint32_t write(cereal::Event::Reader event);

  LoggerdState *s;
  const std::string name;
  std::unique_ptr<VideoWriter> writer;
  int encoderd_segment_offset = -1;
  std::vector<Message *> q;
  int dropped_frames = 0;
  bool recording = false;
  LogCameraInfo cam_info = {};
  cereal::EncodeData::Reader (cereal::Event::Reader::*getEncodeData)() const;
  void (cereal::Event::Builder::*setEncodeIdx)(::cereal::EncodeIndex::Reader value);
};

RemoteEncoder::RemoteEncoder(LoggerdState *s, const std::string &name) : s(s), name(name) {
  if (name == "roadEncodeData") {
    cam_info = cameras_logged[0];
    getEncodeData = &cereal::Event::Reader::getRoadEncodeData;
    setEncodeIdx = &cereal::Event::Builder::setRoadEncodeIdx;
  } else if (name == "driverEncodeData") {
    cam_info = cameras_logged[1];
    getEncodeData = &cereal::Event::Reader::getDriverEncodeData;
    setEncodeIdx = &cereal::Event::Builder::setDriverEncodeIdx;
  } else if (name == "wideRoadEncodeData") {
    cam_info = cameras_logged[2];
    getEncodeData = &cereal::Event::Reader::getWideRoadEncodeData;
    setEncodeIdx = &cereal::Event::Builder::setWideRoadEncodeIdx;
  } else {
    cam_info = qcam_info;
    getEncodeData = &cereal::Event::Reader::getQRoadEncodeData;
    setEncodeIdx = &cereal::Event::Builder::setQRoadEncodeIdx;
  }
}

uint32_t RemoteEncoder::handlePacket(Message *raw_msg) {
  std::unique_ptr<Message> msg(raw_msg);
  capnp::FlatArrayMessageReader cmsg({(capnp::word *)msg->getData(), msg->getSize()});
  const auto event = cmsg.getRoot<cereal::Event>();
  const int32_t segment_num = (event.*getEncodeData)().getIdx().getSegmentNum();

  // encoderd can have started long before loggerd
  if (encoderd_segment_offset == -1) {
    encoderd_segment_offset = segment_num;
    LOGD("%s: has encoderd offset %d", name.c_str(), segment_num);
  }

  const int offset_segment_num = segment_num - encoderd_segment_offset;
  if (offset_segment_num == s->rotate_segment) {
    // loggerd is now on the segment that matches this packet
    return writePacket(event);
  }

  if (offset_segment_num > s->rotate_segment) {
    // encoderd packet has a newer segment, this means encoderd has rolled over
    if (q.empty()) {
      writer.reset();
      recording = false;
      ++s->ready_to_rotate;
      LOGD("rotate %d -> %d ready %d/%d for %s", s->rotate_segment.load(), offset_segment_num, s->ready_to_rotate.load(), s->max_waiting, name.c_str());
    }
    // queue up all the new segment messages, they go in after the rotate
    q.push_back(msg.release());
  } else {
    LOGE("%s: encoderd packet has a older segment!!! idx.getSegmentNum():%d s->rotate_segment:%d encoderd_segment_offset:%d",
         name.c_str(), segment_num, s->rotate_segment.load(), encoderd_segment_offset);
    // this should never happen
    // actually, this can happen if you restart encoderd
    encoderd_segment_offset = -s->rotate_segment.load();
  }
  return 0;
}

uint32_t RemoteEncoder::writePacket(cereal::Event::Reader event) {
  uint32_t bytes_count = 0;
  if (!q.empty()) {
    for (Message *msg : q) {
      capnp::FlatArrayMessageReader cmsg({(capnp::word *)msg->getData(), msg->getSize()});
      bytes_count += write(cmsg.getRoot<cereal::Event>());
      delete msg;
    }
    q.clear();
  }
  bytes_count += write(event);
  return bytes_count;
}

uint32_t RemoteEncoder::write(cereal::Event::Reader event) {
  const auto edata = (event.*getEncodeData)();
  const auto idx = edata.getIdx();
  const auto flags = idx.getFlags();
  if (!recording) {
    if (flags & V4L2_BUF_FLAG_KEYFRAME) {
      // only create on iframe
      if (dropped_frames) {
        // this should only happen for the first segment, maybe
        LOGW("%s: dropped %d non iframe packets before init", name.c_str(), dropped_frames);
        dropped_frames = 0;
      }
      // if we aren't actually recording, don't create the writer
      if (cam_info.record) {
        writer.reset(new VideoWriter(s->segment_path, cam_info.filename, idx.getType() != cereal::EncodeIndex::Type::FULL_H_E_V_C,
                                     cam_info.frame_width, cam_info.frame_height, cam_info.fps, idx.getType()));
        // write the header
        auto header = edata.getHeader();
        writer->write((uint8_t *)header.begin(), header.size(), idx.getTimestampEof() / 1000, true, false);
      }
      recording = true;
    } else {
      ++dropped_frames;
      return 0;
    }
  }

  if (writer) {
    auto data = edata.getData();
    writer->write((uint8_t *)data.begin(), data.size(), idx.getTimestampEof() / 1000, false, flags & V4L2_BUF_FLAG_KEYFRAME);
  }

  // put it in log stream as the idx packet
  MessageBuilder cmsg;
  auto evt = cmsg.initEvent(event.getValid());
  evt.setLogMonoTime(event.getLogMonoTime());
  (evt.*setEncodeIdx)(idx);
  auto new_msg = cmsg.toBytes();
  logger_log(&s->logger, (uint8_t *)new_msg.begin(), new_msg.size(), true);  // always in qlog?
  return new_msg.size();
}

int handle_encoder_msg(LoggerdState *s, Message *msg, const std::string &name) {
  auto &re = s->encoders[name];
  if (!re) {
    re.reset(new RemoteEncoder(s, name));
  }
  s->last_camera_seen_tms = millis_since_boot();
  return re->handlePacket(msg);
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

  // init encoders
  s.last_camera_seen_tms = millis_since_boot();
  for (const auto &cam : cameras_logged) {
    s.max_waiting++;
    if (cam.has_qcamera) { s.max_waiting++; }
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
          bytes_count += handle_encoder_msg(&s, msg, qs.name);
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
