#include <sys/xattr.h>

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/params.h"
#include "system/loggerd/encoder/encoder.h"
#include "system/loggerd/loggerd.h"
#include "system/loggerd/video_writer.h"

ExitHandler do_exit;
std::atomic<bool> dashcam_runtime_fault = false;

void handle_dashcam_runtime_fault(int) {
  dashcam_runtime_fault = true;
  do_exit = true;
}

constexpr double PANDA_STATE_MAX_GAP_MS = 2000.0;

struct LoggerdState {
  LoggerState logger;
  std::atomic<double> last_camera_seen_tms{0.0};
  std::atomic<int> ready_to_rotate{0};  // count of encoders ready to rotate
  int max_waiting = 0;
  double last_rotate_tms = 0.;      // last rotate time in ms
  double last_panda_state_tms = 0.;
  bool panda_state_seen = false;
  bool panda_state_continuous = true;
  bool panda_state_safe = true;
  bool can_seen = false;
};

bool panda_recording_complete(const LoggerdState *s) {
  return s->can_seen && s->panda_state_seen && s->panda_state_continuous && s->panda_state_safe &&
         (millis_since_boot() - s->last_panda_state_tms) <= PANDA_STATE_MAX_GAP_MS;
}

bool panda_state_is_receive_only(cereal::Event::Reader event) {
  const auto panda_states = event.getPandaStates();
  if (panda_states.size() != 1) return false;

  const auto state = panda_states[0];
  if (state.getPandaType() != cereal::PandaState::PandaType::RED_PANDA ||
      state.getSafetyModel() != cereal::CarParams::SafetyModel::SILENT || state.getControlsAllowed() ||
      state.getRxBufferOverflow() != 0) {
    return false;
  }

  const auto can_state_is_receive_only = [](cereal::PandaState::PandaCanState::Reader can_state) {
    return can_state.getTotalTxCnt() == 0 && can_state.getTotalFwdCnt() == 0 && can_state.getTotalRxLostCnt() == 0;
  };
  return can_state_is_receive_only(state.getCanState0()) &&
         can_state_is_receive_only(state.getCanState1()) &&
         can_state_is_receive_only(state.getCanState2());
}

void logger_rotate(LoggerdState *s, bool previous_segment_complete = true) {
  bool ret = s->logger.next(previous_segment_complete);
  assert(ret);
  s->ready_to_rotate = 0;
  s->last_rotate_tms = millis_since_boot();
  s->last_panda_state_tms = 0.;
  s->panda_state_seen = false;
  s->panda_state_continuous = true;
  s->panda_state_safe = true;
  s->can_seen = false;
  LOGW((s->logger.segment() == 0) ? "logging to %s" : "rotated to %s", s->logger.segmentPath().c_str());
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
    // Keep the route-level lock unless both encoders and the passive Panda data
    // path stayed healthy throughout the old segment.
    const bool panda_complete = getenv("DASHCAM") == nullptr || panda_recording_complete(s);
    logger_rotate(s, all_ready && panda_complete);
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
  bool audio_initialized = false;
};

size_t write_encode_data(LoggerdState *s, cereal::Event::Reader event, RemoteEncoder &re, const EncoderInfo &encoder_info) {
  auto edata = (event.*(encoder_info.get_encode_data_func))();
  auto idx = edata.getIdx();
  auto flags = idx.getFlags();

  // if we aren't recording yet, try to start, since we are in the correct segment
  if (!re.recording) {
    if (flags & V4L2_BUF_FLAG_KEYFRAME) {
      // only create on iframe
      if (re.dropped_frames) {
        // this should only happen for the first segment, maybe
        LOGW("%s: dropped %d non iframe packets before init", encoder_info.publish_name, re.dropped_frames);
        re.dropped_frames = 0;
      }
      if (encoder_info.record) {
        // write the header
        auto header = edata.getHeader();
        re.writer->write((uint8_t *)header.begin(), header.size(), idx.getTimestampEof() / 1000, true, false);
      }
      re.recording = true;
    } else {
      // this is a sad case when we aren't recording, but don't have an iframe
      // nothing we can do but drop the frame
      ++re.dropped_frames;
      return 0;
    }
  }

  // we have to be recording if we are here
  assert(re.recording);

  // if we are actually writing the video file, do so
  if (re.writer) {
    auto data = edata.getData();
    re.writer->write((uint8_t *)data.begin(), data.size(), idx.getTimestampEof() / 1000, false, flags & V4L2_BUF_FLAG_KEYFRAME);
  }

  // put it in log stream as the idx packet
  MessageBuilder bmsg;
  auto evt = bmsg.initEvent(event.getValid());
  evt.setLogMonoTime(event.getLogMonoTime());
  (evt.*(encoder_info.set_encode_idx_func))(idx);
  auto new_msg = bmsg.toBytes();
  s->logger.write((uint8_t *)new_msg.begin(), new_msg.size(), true);  // always in qlog?
  return new_msg.size();
}

int handle_encoder_msg(LoggerdState *s, Message *msg, std::string &name, struct RemoteEncoder &re, const EncoderInfo &encoder_info) {
  int bytes_count = 0;

  // extract the message
  capnp::FlatArrayMessageReader cmsg(kj::ArrayPtr<capnp::word>((capnp::word *)msg->getData(), msg->getSize() / sizeof(capnp::word)));
  auto event = cmsg.getRoot<cereal::Event>();
  auto edata = (event.*(encoder_info.get_encode_data_func))();
  auto idx = edata.getIdx();

  // encoderd can have started long before loggerd
  if (!re.seen_first_packet) {
    re.seen_first_packet = true;
    re.encoderd_segment_offset = idx.getSegmentNum();
    LOGD("%s: has encoderd offset %d", name.c_str(), re.encoderd_segment_offset);
  }
  int offset_segment_num = idx.getSegmentNum() - re.encoderd_segment_offset;

  if (offset_segment_num == s->logger.segment()) {
    // loggerd is now on the segment that matches this packet

    // if this is a new segment, we close any possible old segments, move to the new, and process any queued packets
    if (re.current_segment != s->logger.segment()) {
      // if we aren't actually recording, don't create the writer
      if (encoder_info.record) {
        assert(encoder_info.filename != NULL);
        re.writer.reset(new VideoWriter(s->logger.segmentPath().c_str(),
                                        encoder_info.filename, idx.getType() != cereal::EncodeIndex::Type::FULL_H_E_V_C,
                                        edata.getWidth(), edata.getHeight(), encoder_info.fps, idx.getType()));
        re.recording = false;
        re.audio_initialized = false;
      }
      re.current_segment = s->logger.segment();
      re.marked_ready_to_rotate = false;
    }
    if (re.audio_initialized || !encoder_info.include_audio) {
      // we are in this segment now, process any queued messages before this one
      if (!re.q.empty()) {
        for (auto qmsg : re.q) {
          capnp::FlatArrayMessageReader reader({(capnp::word *)qmsg->getData(), qmsg->getSize() / sizeof(capnp::word)});
          bytes_count += write_encode_data(s, reader.getRoot<cereal::Event>(), re, encoder_info);
          delete qmsg;
        }
        re.q.clear();
      }
      bytes_count += write_encode_data(s, event, re, encoder_info);
      delete msg;
    } else if (re.q.size() > MAIN_FPS*10) {
      LOGE_100("%s: dropping frame waiting for audio initialization, queue is too large", name.c_str());
      delete msg;
    } else {
      re.q.push_back(msg); // queue up all the new segment messages, they go in after audio is initialized
    }
  } else if (offset_segment_num > s->logger.segment()) {
    // encoderd packet has a newer segment, this means encoderd has rolled over
    if (!re.marked_ready_to_rotate) {
      re.marked_ready_to_rotate = true;
      ++s->ready_to_rotate;
      LOGD("rotate %d -> %d ready %d/%d for %s",
        s->logger.segment(), offset_segment_num,
        s->ready_to_rotate.load(), s->max_waiting, name.c_str());
    }

    // TODO: define this behavior, but for now don't leak
    if (re.q.size() > MAIN_FPS*10) {
      LOGE_100("%s: dropping frame, queue is too large", name.c_str());
      delete msg;
    } else {
      // queue up all the new segment messages, they go in after the rotate
      re.q.push_back(msg);
    }
  } else {
    LOGE("%s: encoderd packet has a older segment!!! idx.getSegmentNum():%d s->logger.segment():%d re.encoderd_segment_offset:%d",
      name.c_str(), idx.getSegmentNum(), s->logger.segment(), re.encoderd_segment_offset);
    // free the message, it's useless. this should never happen
    // actually, this can happen if you restart encoderd
    re.encoderd_segment_offset = -s->logger.segment();
    delete msg;
  }

  return bytes_count;
}

void handle_preserve_segment(LoggerdState *s) {
  static int prev_segment = -1;
  if (s->logger.segment() == prev_segment) return;

  LOGW("preserving %s", s->logger.segmentPath().c_str());

#ifdef __APPLE__
  int ret = setxattr(s->logger.segmentPath().c_str(), PRESERVE_ATTR_NAME, &PRESERVE_ATTR_VALUE, 1, 0, 0);
#else
  int ret = setxattr(s->logger.segmentPath().c_str(), PRESERVE_ATTR_NAME, &PRESERVE_ATTR_VALUE, 1, 0);
#endif
  if (ret) {
    LOGE("setxattr %s failed for %s: %s", PRESERVE_ATTR_NAME, s->logger.segmentPath().c_str(), strerror(errno));
  }

  // mark route for uploading
  Params params;
  std::string routes = params.get("AthenadRecentlyViewedRoutes");
  params.put("AthenadRecentlyViewedRoutes", routes + "," + s->logger.routeName());

  prev_segment = s->logger.segment();
}

void loggerd_thread() {
  const bool dashcam = getenv("DASHCAM") != nullptr;

  // setup messaging
  struct ServiceState {
    std::string name;
    int counter, freq;
    bool encoder, preserve_segment, record_audio;
  };
  std::unordered_map<SubSocket*, ServiceState> service_state;
  std::unordered_map<SubSocket*, struct RemoteEncoder> remote_encoders;

  std::unique_ptr<Context> ctx(Context::create());
  std::unique_ptr<Poller> poller(Poller::create());

  // subscribe to all socks
  for (const auto& [_, it] : services) {
    const bool encoder = util::ends_with(it.name, "EncodeData");
    const bool livestream_encoder = util::starts_with(it.name, "livestream");
    const bool record_audio = (it.name == "rawAudioData") && Params().getBool("RecordAudio");
    if (it.should_log || (encoder && !livestream_encoder) || record_audio) {
      LOGD("logging %s", it.name.c_str());

      SubSocket * sock = SubSocket::create(ctx.get(), it.name, "127.0.0.1", false, true, it.queue_size);
      assert(sock != NULL);
      poller->registerSocket(sock);
      service_state[sock] = {
        .name = it.name,
        .counter = 0,
        .freq = it.decimation,
        .encoder = encoder,
        .preserve_segment = (it.name == "userBookmark") || (it.name == "audioFeedback"),
        .record_audio = record_audio,
      };
    }
  }

  LoggerdState s;
  // init logger
  logger_rotate(&s);
  Params().put("CurrentRoute", s.logger.routeName());

  std::map<std::string, EncoderInfo> encoder_infos_dict;
  std::vector<RemoteEncoder*> encoders_with_audio;
  for (const auto &cam : cameras_logged) {
    // The CM5 baseline exposes only the road stream. Count only its main and
    // qcamera encoders so loggerd rotates in lockstep with encoderd at 60 s.
    if (dashcam && cam.stream_type != VISION_STREAM_ROAD) continue;
    for (const auto &encoder_info : cam.encoder_infos) {
      encoder_infos_dict[encoder_info.publish_name] = encoder_info;
      s.max_waiting++;
    }
  }

  for (auto &[sock, service] : service_state) {
    auto it = encoder_infos_dict.find(service.name);
    if (it != encoder_infos_dict.end() && it->second.include_audio) {
      encoders_with_audio.push_back(&remote_encoders[sock]);
    }
  }

  uint64_t msg_count = 0, bytes_count = 0;
  double start_ts = millis_since_boot();
  while (!do_exit) {
    // poll for new messages on all sockets
    for (auto sock : poller->poll(1000)) {
      if (do_exit) break;

      ServiceState &service = service_state[sock];
      if (service.preserve_segment) {
        handle_preserve_segment(&s);
      }

      // drain socket
      int count = 0;
      Message *msg = nullptr;
      while (!do_exit && (msg = sock->receive(true))) {
        const bool in_qlog = service.freq != -1 && (service.counter++ % service.freq == 0);

        if (dashcam && service.name == "pandaStates") {
          const double now = millis_since_boot();
          const double previous = s.panda_state_seen ? s.last_panda_state_tms : s.last_rotate_tms;
          if ((now - previous) > PANDA_STATE_MAX_GAP_MS) s.panda_state_continuous = false;
          capnp::FlatArrayMessageReader cmsg(kj::ArrayPtr<capnp::word>((capnp::word *)msg->getData(), msg->getSize() / sizeof(capnp::word)));
          if (!panda_state_is_receive_only(cmsg.getRoot<cereal::Event>())) s.panda_state_safe = false;
          s.panda_state_seen = true;
          s.last_panda_state_tms = now;
        } else if (dashcam && service.name == "can") {
          // The dedicated USB publisher suppresses empty CAN batches.
          s.can_seen = true;
        }

        if (service.record_audio) {
          capnp::FlatArrayMessageReader cmsg(kj::ArrayPtr<capnp::word>((capnp::word *)msg->getData(), msg->getSize() / sizeof(capnp::word)));
          auto event = cmsg.getRoot<cereal::Event>();
          auto audio_data = event.getRawAudioData().getData();
          auto sample_rate = event.getRawAudioData().getSampleRate();
          for (auto* encoder : encoders_with_audio) {
            if (encoder && encoder->writer) {
              encoder->writer->write_audio((uint8_t*)audio_data.begin(), audio_data.size(), event.getLogMonoTime() / 1000, sample_rate);
              encoder->audio_initialized = true;
            }
          }
        }

        if (service.encoder) {
          s.last_camera_seen_tms = millis_since_boot();
          bytes_count += handle_encoder_msg(&s, msg, service.name, remote_encoders[sock], encoder_infos_dict[service.name]);
        } else {
          s.logger.write((uint8_t *)msg->getData(), msg->getSize(), in_qlog);
          bytes_count += msg->getSize();
          delete msg;
        }

        rotate_if_needed(&s);

        if ((++msg_count % 10000) == 0) {
          double seconds = (millis_since_boot() - start_ts) / 1000.0;
          LOGD("%" PRIu64 " messages, %.2f msg/sec, %.2f KB/sec", msg_count, msg_count / seconds, bytes_count * 0.001 / seconds);
        }

        count++;
        if (count >= 200) {
          LOGD("large volume of '%s' messages", service.name.c_str());
          break;
        }
      }
    }
  }

  LOGW("closing logger");
  if (dashcam_runtime_fault && !s.logger.mark_previous_segment_incomplete()) {
    LOGE("failed to re-lock previous segment after dashcam runtime fault");
  }
  // Snapshot input health before potentially slow video trailer/fsync work.
  const bool panda_complete = !dashcam || panda_recording_complete(&s);

  // Finalize and durably close every video before declaring the segment logs
  // complete. Any failed video finalization deliberately leaves its .lock.
  size_t expected_video_writers = 0;
  size_t finalized_video_writers = 0;
  if (dashcam) {
    for (const auto &[sock, service] : service_state) {
      auto encoder_info = encoder_infos_dict.find(service.name);
      if (encoder_info == encoder_infos_dict.end() || !encoder_info->second.record) continue;

      ++expected_video_writers;
      auto encoder = remote_encoders.find(sock);
      if (encoder != remote_encoders.end() && encoder->second.current_segment == s.logger.segment() &&
          encoder->second.writer && encoder->second.writer->close()) {
        ++finalized_video_writers;
      }
    }
  }
  for (auto &[sock, encoder] : remote_encoders) {
    encoder.writer.reset();
  }
  const bool videos_complete = !dashcam || (expected_video_writers > 0 && finalized_video_writers == expected_video_writers);
  if (!videos_complete) {
    LOGE("keeping route lock: finalized %zu/%zu expected video writers", finalized_video_writers, expected_video_writers);
  }
  if (!panda_complete) {
    LOGE("keeping route lock: CAN/Panda health was missing, discontinuous, or not receive-only");
  }
  const bool segment_complete = videos_complete && panda_complete;
  if (dashcam_runtime_fault) {
    LOGE("keeping route lock: dashcam supervisor reported a runtime fault");
  }
  // Read the exit signal after potentially slow video finalization so a late
  // SIGPWR upgrade is reflected in the ending sentinel.
  s.logger.setExitSignal(do_exit.signal);
  s.logger.close(dashcam || do_exit.power_failure, segment_complete && !dashcam_runtime_fault);

  if (do_exit.power_failure) {
    LOGE("power failure");
    // Logger and video writers have already emitted their final sentinel or
    // trailer and fsync'd. This final sync covers the remaining system state.
    sync();
    LOGE("sync done");
  }

  // messaging cleanup
  for (auto &[sock, service] : service_state) delete sock;
}

int main(int argc, char** argv) {
  std::signal(SIGUSR1, handle_dashcam_runtime_fault);
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
