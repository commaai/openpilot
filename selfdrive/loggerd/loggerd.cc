#include "selfdrive/loggerd/loggerd.h"

ExitHandler do_exit;

// Handle initial encoder syncing by waiting for all encoders to reach the same frame id
bool sync_encoders(LoggerdState *s, CameraType cam_type, uint32_t frame_id) {
  if (s->camera_synced[cam_type]) return true;

  if (s->max_waiting > 1 && s->encoders_ready != s->max_waiting) {
    // add a small margin to the start frame id in case one of the encoders already dropped the next frame
    update_max_atomic(s->start_frame_id, frame_id + 2);
    if (std::exchange(s->camera_ready[cam_type], true) == false) {
      ++s->encoders_ready;
      LOGE("camera %d encoder ready", cam_type);
    }
    return false;
  } else {
    if (s->max_waiting == 1) update_max_atomic(s->start_frame_id, frame_id);
    bool synced = frame_id >= s->start_frame_id;
    s->camera_synced[cam_type] = synced;
    if (!synced) LOGE("camera %d waiting for frame %d, cur %d", cam_type, (int)s->start_frame_id, frame_id);
    return synced;
  }
}

bool trigger_rotate_if_needed(LoggerdState *s, int cur_seg, uint32_t frame_id) {
  const int frames_per_seg = SEGMENT_LENGTH * MAIN_FPS;
  if (cur_seg >= 0 && frame_id >= ((cur_seg + 1) * frames_per_seg) + s->start_frame_id) {
    // trigger rotate and wait until the main logger has rotated to the new segment
    ++s->ready_to_rotate;
    std::unique_lock lk(s->rotate_lock);
    s->rotate_cv.wait(lk, [&] {
      return s->rotate_segment > cur_seg || do_exit;
    });
    return !do_exit;
  }
  return false;
}

void encoder_thread(LoggerdState *s, const LogCameraInfo &cam_info) {
  set_thread_name(cam_info.filename);

  int cur_seg = -1;
  int encode_idx = 0;
  LoggerHandle *lh = NULL;
  std::vector<Encoder *> encoders;
  VisionIpcClient vipc_client = VisionIpcClient("camerad", cam_info.stream_type, false);

  while (!do_exit) {
    if (!vipc_client.connect(false)) {
      util::sleep_for(5);
      continue;
    }

    // init encoders
    if (encoders.empty()) {
      VisionBuf buf_info = vipc_client.buffers[0];
      LOGD("encoder init %dx%d", buf_info.width, buf_info.height);

      // main encoder
      encoders.push_back(new Encoder(cam_info.filename, buf_info.width, buf_info.height,
                                     cam_info.fps, cam_info.bitrate, cam_info.is_h265,
                                     cam_info.downscale, cam_info.record));
      // qcamera encoder
      if (cam_info.has_qcamera) {
        encoders.push_back(new Encoder(qcam_info.filename, qcam_info.frame_width, qcam_info.frame_height,
                                       qcam_info.fps, qcam_info.bitrate, qcam_info.is_h265, qcam_info.downscale));
      }
    }

    while (!do_exit) {
      VisionIpcBufExtra extra;
      VisionBuf* buf = vipc_client.recv(&extra);
      if (buf == nullptr) continue;

      if (cam_info.trigger_rotate) {
        s->last_camera_seen_tms = millis_since_boot();
        if (!sync_encoders(s, cam_info.type, extra.frame_id)) {
          continue;
        }

        // check if we're ready to rotate
        trigger_rotate_if_needed(s, cur_seg, extra.frame_id);
        if (do_exit) break;
      }

      // rotate the encoder if the logger is on a newer segment
      if (s->rotate_segment > cur_seg) {
        cur_seg = s->rotate_segment;

        LOGW("camera %d rotate encoder to %s", cam_info.type, s->segment_path);
        for (auto &e : encoders) {
          e->encoder_close();
          e->encoder_open(s->segment_path);
        }
        if (lh) {
          lh_close(lh);
        }
        lh = logger_get_handle(&s->logger);
      }

      // encode a frame
      for (int i = 0; i < encoders.size(); ++i) {
        int out_id = encoders[i]->encode_frame(buf->y, buf->u, buf->v,
                                               buf->width, buf->height, extra.timestamp_eof);

        if (out_id == -1) {
          LOGE("Failed to encode frame. frame_id: %d encode_id: %d", extra.frame_id, encode_idx);
        }

        // publish encode index
        if (i == 0 && out_id != -1) {
          MessageBuilder msg;
          // this is really ugly
          bool valid = (buf->get_frame_id() == extra.frame_id);
          auto eidx = cam_info.type == DriverCam ? msg.initEvent(valid).initDriverEncodeIdx() :
                     (cam_info.type == WideRoadCam ? msg.initEvent(valid).initWideRoadEncodeIdx() : msg.initEvent(valid).initRoadEncodeIdx());
          eidx.setFrameId(extra.frame_id);
          eidx.setTimestampSof(extra.timestamp_sof);
          eidx.setTimestampEof(extra.timestamp_eof);
          if (Hardware::TICI()) {
            eidx.setType(cereal::EncodeIndex::Type::FULL_H_E_V_C);
          } else {
            eidx.setType(cam_info.type == DriverCam ? cereal::EncodeIndex::Type::FRONT : cereal::EncodeIndex::Type::FULL_H_E_V_C);
          }
          eidx.setEncodeId(encode_idx);
          eidx.setSegmentNum(cur_seg);
          eidx.setSegmentId(out_id);
          if (lh) {
            auto bytes = msg.toBytes();
            lh_log(lh, bytes.begin(), bytes.size(), true);
          }
        }
      }

      encode_idx++;
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
  s->rotate_cv.notify_all();
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

void loggerd_thread() {
  // setup messaging
  typedef struct QlogState {
    std::string name;
    int counter, freq;
  } QlogState;
  std::unordered_map<SubSocket*, QlogState> qlog_states;

  std::unique_ptr<Context> ctx(Context::create());
  std::unique_ptr<Poller> poller(Poller::create());

  // subscribe to all socks
  for (const auto& it : services) {
    if (!it.should_log) continue;

    SubSocket * sock = SubSocket::create(ctx.get(), it.name);
    assert(sock != NULL);
    poller->registerSocket(sock);
    qlog_states[sock] = {
      .name = it.name,
      .counter = 0,
      .freq = it.decimation,
    };
  }

  LoggerdState s;
  // init logger
  logger_init(&s.logger, "rlog", true);
  logger_rotate(&s);
  Params().put("CurrentRoute", s.logger.route_name);

  // init encoders
  s.last_camera_seen_tms = millis_since_boot();
  std::vector<std::thread> encoder_threads;
  for (const auto &cam : cameras_logged) {
    if (cam.enable) {
      encoder_threads.push_back(std::thread(encoder_thread, &s, cam));
      if (cam.trigger_rotate) s.max_waiting++;
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
        logger_log(&s.logger, (uint8_t *)msg->getData(), msg->getSize(), in_qlog);
        bytes_count += msg->getSize();
        delete msg;

        rotate_if_needed(&s);

        if ((++msg_count % 1000) == 0) {
          double seconds = (millis_since_boot() - start_ts) / 1000.0;
          LOGD("%lu messages, %.2f msg/sec, %.2f KB/sec", msg_count, msg_count / seconds, bytes_count * 0.001 / seconds);
        }

        count++;
        if (count >= 200) {
          LOGE("large volume of '%s' messages", qs.name.c_str());
          break;
        }
      }
    }
  }

  LOGW("closing encoders");
  s.rotate_cv.notify_all();
  for (auto &t : encoder_threads) t.join();

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
