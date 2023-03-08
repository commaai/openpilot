#include "system/loggerd/loggerd.h"
#include "system/loggerd/video_writer.h"

ExitHandler do_exit;

struct LoggerdState {
  LoggerState logger = {};
  char segment_path[4096];
  std::atomic<int> rotate_segment;
  std::atomic<double> last_camera_seen_tms;
  std::atomic<int> ready_to_rotate;  // count of encoders ready to rotate
  int max_waiting = 0;
  double last_rotate_tms = 0.;      // last rotate time in ms
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
  // all encoders ready, trigger rotation
  bool all_ready = s->ready_to_rotate == s->max_waiting;

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

struct RemoteEncoder {
  std::unique_ptr<VideoWriter> writer;
  int encoderd_segment_offset;
  int current_segment = -1;
  std::vector<Message *> q;
  int dropped_frames = 0;
  bool recording = false;
  bool marked_ready_to_rotate = false;
  bool seen_first_packet = false;
};

int handle_encoder_msg(LoggerdState *s, Message *msg, std::string &name, struct RemoteEncoder &re) {
  const LogCameraInfo &cam_info = (name == "driverEncodeData") ? cameras_logged[1] :
    ((name == "wideRoadEncodeData") ? cameras_logged[2] :
    ((name == "qRoadEncodeData") ? qcam_info : cameras_logged[0]));
  int bytes_count = 0;

  // extract the message
  capnp::FlatArrayMessageReader cmsg(kj::ArrayPtr<capnp::word>((capnp::word *)msg->getData(), msg->getSize() / sizeof(capnp::word)));
  auto event = cmsg.getRoot<cereal::Event>();
  auto edata = (name == "driverEncodeData") ? event.getDriverEncodeData() :
    ((name == "wideRoadEncodeData") ? event.getWideRoadEncodeData() :
    ((name == "qRoadEncodeData") ? event.getQRoadEncodeData() : event.getRoadEncodeData()));
  auto idx = edata.getIdx();
  auto flags = idx.getFlags();

  // encoderd can have started long before loggerd
  if (!re.seen_first_packet) {
    re.seen_first_packet = true;
    re.encoderd_segment_offset = idx.getSegmentNum();
    LOGD("%s: has encoderd offset %d", name.c_str(), re.encoderd_segment_offset);
  }
  int offset_segment_num = idx.getSegmentNum() - re.encoderd_segment_offset;

  if (offset_segment_num == s->rotate_segment) {
    // loggerd is now on the segment that matches this packet

    // if this is a new segment, we close any possible old segments, move to the new, and process any queued packets
    if (re.current_segment != s->rotate_segment) {
      if (re.recording) {
        re.writer.reset();
        re.recording = false;
      }
      re.current_segment = s->rotate_segment;
      re.marked_ready_to_rotate = false;
      // we are in this segment now, process any queued messages before this one
      if (!re.q.empty()) {
        for (auto &qmsg: re.q) {
          bytes_count += handle_encoder_msg(s, qmsg, name, re);
        }
        re.q.clear();
      }
    }

    // if we aren't recording yet, try to start, since we are in the correct segment
    if (!re.recording) {
      if (flags & V4L2_BUF_FLAG_KEYFRAME) {
        // only create on iframe
        if (re.dropped_frames) {
          // this should only happen for the first segment, maybe
          LOGW("%s: dropped %d non iframe packets before init", name.c_str(), re.dropped_frames);
          re.dropped_frames = 0;
        }
        // if we aren't actually recording, don't create the writer
        if (cam_info.record) {
          re.writer.reset(new VideoWriter(s->segment_path,
            cam_info.filename, idx.getType() != cereal::EncodeIndex::Type::FULL_H_E_V_C,
            cam_info.frame_width, cam_info.frame_height, cam_info.fps, idx.getType()));
          // write the header
          auto header = edata.getHeader();
          re.writer->write((uint8_t *)header.begin(), header.size(), idx.getTimestampEof()/1000, true, false);
        }
        re.recording = true;
      } else {
        // this is a sad case when we aren't recording, but don't have an iframe
        // nothing we can do but drop the frame
        delete msg;
        ++re.dropped_frames;
        return bytes_count;
      }
    }

    // we have to be recording if we are here
    assert(re.recording);

    // if we are actually writing the video file, do so
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

    // free the message, we used it
    delete msg;
  } else if (offset_segment_num > s->rotate_segment) {
    // encoderd packet has a newer segment, this means encoderd has rolled over
    if (!re.marked_ready_to_rotate) {
      re.marked_ready_to_rotate = true;
      ++s->ready_to_rotate;
      LOGD("rotate %d -> %d ready %d/%d for %s",
        s->rotate_segment.load(), offset_segment_num,
        s->ready_to_rotate.load(), s->max_waiting, name.c_str());
    }
    // queue up all the new segment messages, they go in after the rotate
    re.q.push_back(msg);
  } else {
    LOGE("%s: encoderd packet has a older segment!!! idx.getSegmentNum():%d s->rotate_segment:%d re.encoderd_segment_offset:%d",
      name.c_str(), idx.getSegmentNum(), s->rotate_segment.load(), re.encoderd_segment_offset);
    // free the message, it's useless. this should never happen
    // actually, this can happen if you restart encoderd
    re.encoderd_segment_offset = -s->rotate_segment.load();
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
