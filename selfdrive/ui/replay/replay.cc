#include "selfdrive/ui/replay/replay.h"

#include <capnp/dynamic.h>

#include "cereal/services.h"
#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/hardware/hw.h"

const int SEGMENT_LENGTH = 60;  // 60s
const int FORWARD_SEGS = 2;
const int BACKWARD_SEGS = 2;

// class Replay

Replay::Replay(SubMaster *sm, QObject *parent) : sm_(sm), QObject(parent) {
  QStringList block = QString(getenv("BLOCK")).split(",");
  qDebug() << "blocklist" << block;
  QStringList allow = QString(getenv("ALLOW")).split(",");
  qDebug() << "allowlist" << allow;

  std::vector<const char *> s;
  for (const auto &it : services) {
    if ((allow[0].size() == 0 || allow.contains(it.name)) &&
        !block.contains(it.name)) {
      s.push_back(it.name);
      socks_.insert(it.name);
    }
  }
  qDebug() << "services " << s;

  if (sm_ == nullptr) {
    pm_ = new PubMaster(s);
  }
  events_ = new std::vector<Event *>();
}

Replay::~Replay() {
  stop();
  delete pm_;
}

bool Replay::start(const QString &routeName) {
  Route route(routeName);
  if (!route.load()) {
    qInfo() << "failed to retrieve files for route " << routeName;
    return false;
  }
  return start(route);
}

bool Replay::start(const Route &route) {
  assert(!running());
  if (!route.segments().size()) return false;

  route_ = route;
  current_segment_ = route_.segments().firstKey();

  qDebug() << "replay route " << route_.name() << " from " << current_segment_ << ", total segments:" << route.segments().size();
  queue_thread_ = std::thread(&Replay::queueSegmentThread, this);
  stream_thread_ = std::thread(&Replay::streamThread, this);
  return true;
}

void Replay::stop() {
  if (!running()) return;

  // wait until threads finished
  camera_server_.stop();
  exit_ = true;
  stream_thread_.join();
  queue_thread_.join();
  exit_ = false;

  segments_.clear();
  for (auto cam_type : ALL_CAMERAS) {
    encoderIdx_[cam_type].clear();
  }
  for (Event *e : *events_) {
    delete e;
  }
  events_->clear();

  current_ts_ = seek_ts_ = 0;
  current_segment_ = 0;
  route_start_ts_ = 0;
}

QString Replay::elapsedTime(uint64_t ns) {
  QTime time(0, 0, 0);
  auto a = time.addSecs((ns - route_start_ts_) / 1e9);
  return a.toString("hh:mm:ss");
}

void Replay::relativeSeek(int seconds) {
  seekTo(current_ts_ + seconds * 1e9);
}

void Replay::seek(int seconds) {
  seekTo(route_start_ts_ + seconds * 1e9);
}

void Replay::seekTo(uint64_t to_ts) {
  const auto &rs = route_.segments();
  if (!rs.size()) return;

  std::unique_lock lk(mutex_);
  seek_ts_ = to_ts;
  int seconds = (to_ts - route_start_ts_) / 1e9;
  current_segment_ = std::clamp(seconds / SEGMENT_LENGTH, 0, rs.lastKey());
  events_changed_ = true;
  qDebug() << "seeking to " << elapsedTime(to_ts);
}

// return nullptr if segment is not loaded
std::shared_ptr<Segment> Replay::getSegment(int segment) {
  auto it = segments_.find(segment);
  return (it != segments_.end() && !it->second->loading()) ? it->second : nullptr;
}

void Replay::pushFrame(CameraType cam_type, uint32_t frame_id) {
  if (!camera_server_.hasCamera(cam_type)) return;

  auto eidx = encoderIdx_[cam_type].find(frame_id);
  if (eidx != encoderIdx_[cam_type].end()) {
    auto seg = getSegment(eidx->second.segmentNum);
    if (seg) {
      camera_server_.pushFrame(cam_type, seg, eidx->second.segmentId);
    }
  }
}

const std::string &Replay::eventSocketName(const cereal::Event::Reader &e) {
  auto it = eventNameMap.find(e.which());
  if (it == eventNameMap.end()) {
    std::string type;
    KJ_IF_MAYBE(e_, static_cast<capnp::DynamicStruct::Reader>(e).which()) {
      type = e_->getProto().getName();
    }
    if (socks_.find(type) == socks_.end()) {
      type = "";
    }
    it = eventNameMap.insert(it, {e.which(), type});
  }
  return it->second;
}

// maintain the segment window
void Replay::queueSegmentThread() {
  static int prev_segment = -1;
  while (!exit_) {
    int segment = current_segment_;
    if (prev_segment == segment) {
      QThread::msleep(100);
      continue;
    }

    const auto &rs = route_.segments();
    const int cur_idx = std::distance(rs.begin(), rs.find(segment));
    int i = 0;
    for (auto it = rs.begin(); it != rs.end(); ++it, ++i) {
      if (i >= cur_idx - BACKWARD_SEGS && i <= cur_idx + FORWARD_SEGS) {
        int n = it.key();
        if (segments_.find(n) == segments_.end()) {
          std::unique_lock lk(mutex_);
          segments_[n] = std::make_shared<Segment>(n, rs[n]);
          connect(segments_[n].get(), &Segment::loaded, [=] {
            mergeEvents(segments_[n]->log);
          });
        }
      }
    }
    prev_segment = segment;
  }
}

// std::vector<Event *>::iterator Replay::currentEvent() {
//   std::vector<Event *>::iterator eit;
//   // std::unique_lock lk(mutex_);
//   if (events_changed_ && current_ts_ > route_start_ts_) {
//     // Make sure not to send duplicate events
//     eit = std::upper_bound(events_->begin(), events_->end(), current_ts_, [&](uint64_t v, const Event *e) {
//       return v < e->mono_time || (v == e->mono_time && which < e->which);
//     });
//   } else {
//     // seeking
//     eit = std::upper_bound(events_->begin(), events_->end(), seek_ts_, [](uint64_t v, const Event *e) {
//       return v < e->mono_time;
//     });
//   }
//   events_changed_ = false;
//   std::shared_ptr<Segment> seg = ;
//   if (eit == events_->end() || (seg = getSegment(current_segment_)) == nullptr) {
//     return events_->end();
//   }
//   return eit;
// }

void Replay::streamThread() {
  uint64_t last_print = 0;
  uint64_t prev_seek_ts = 0;
  cereal::Event::Which which;

  while (!exit_) {
    // double t1 = millis_since_boot();
    mutex_.lock();
    std::shared_ptr<Segment> seg = getSegment(current_segment_);
    uint64_t search_ts = current_ts_;
    if (prev_seek_ts != seek_ts_) {
      which = cereal::Event::INIT_DATA;  // 0
      search_ts = prev_seek_ts = seek_ts_;
    }
    auto eit = std::upper_bound(events_->begin(), events_->end(), search_ts, [&](uint64_t v, const Event *e) {
      return v < e->mono_time || (v == e->mono_time && which < e->which);
    });
    if (!seg || eit == events_->end()) {
      mutex_.unlock();
      qDebug() << "waiting for events";
      QThread::msleep(100);
      continue;
    }
    events_changed_ = false;
    camera_server_.ensureServerForSegment(seg.get());
    mutex_.unlock();

    uint64_t evt_start_tm = (*eit)->mono_time;
    // TODO: use initData's logMonoTime
    if (route_start_ts_ == 0) {
      route_start_ts_ = evt_start_tm;
    }
    uint64_t loop_start_tm = nanos_since_boot();
    while (!exit_) {
      std::unique_lock lk(mutex_);
      if (events_changed_ || eit == events_->end()) break;

      Event *evt = (*eit);
      current_ts_ = evt->mono_time;
      which = evt->which;
      const std::string &sock_name = eventSocketName(evt->event);
      if (!sock_name.empty()) {
        if ((current_ts_ - last_print) > 5 * 1e9) {
          last_print = current_ts_;
          qInfo() << "at segment " << current_segment_ << ": " << elapsedTime(last_print);
        }

        // keep time
        long etime = current_ts_ - evt_start_tm;
        long rtime = nanos_since_boot() - loop_start_tm;
        long us_behind = ((etime - rtime) * 1e-3) + 0.5;
        if (us_behind > 0 && us_behind < 1e6) {
          QThread::usleep(us_behind);
          //qDebug() << "sleeping" << us_behind << etime << timer.nsecsElapsed();
        }

        // publish frames
        switch (which) {
          case cereal::Event::ROAD_CAMERA_STATE:
            pushFrame(RoadCam, evt->event.getRoadCameraState().getFrameId());
            break;
          case cereal::Event::DRIVER_CAMERA_STATE:
            pushFrame(DriverCam, evt->event.getDriverCameraState().getFrameId());
            break;
          case cereal::Event::WIDE_ROAD_CAMERA_STATE:
            pushFrame(WideRoadCam, evt->event.getWideRoadCameraState().getFrameId());
            break;
          default:
            break;
        }

        // publish msg
        if (sm_ == nullptr) {
          auto bytes = evt->bytes();
          pm_->send(sock_name.c_str(), (capnp::byte *)bytes.begin(), bytes.size());
        } else {
          // TODO: subMaster is not thread safe.
          sm_->update_msgs(nanos_since_boot(), {{sock_name, evt->event}});
        }
      }
      current_segment_ = (current_ts_ - route_start_ts_) / 1e9 / SEGMENT_LENGTH;
      ++eit;
    }
  }
}

void Replay::mergeEvents(LogReader *log) {
  // double t1 = millis_since_boot();
  // remove segments
  std::unique_lock lk(merge_mutex_);
  // qInfo() << "mergeEvents start log size " << log->events.size() << " my size " << events_->size();
  uint64_t max_tm = route_start_ts_ + (current_segment_ - BACKWARD_SEGS) * SEGMENT_LENGTH * 1e9;
  uint64_t min_tm = route_start_ts_ + (current_segment_ + FORWARD_SEGS + 1) * SEGMENT_LENGTH * 1e9;
  auto begin_it = std::lower_bound(events_->begin(), events_->end(), max_tm, [](const Event *e, uint64_t v) {
    return e->mono_time < v;
  });
  if (begin_it == events_->end()) {
    begin_it = events_->begin();
  }
  for (auto it = events_->begin(); it != begin_it; ++it) {
    delete *it;
    *it = nullptr;
  }

  auto end_it = std::upper_bound(begin_it, events_->end(), min_tm, [](uint64_t v, const Event *e) {
    return v < e->mono_time;
  });
  for (auto it = end_it; it != events_->end(); ++it) {
    delete *it;
    *it = nullptr;
  }

  // merge segment
  std::vector<Event *> *dst = new std::vector<Event *>;
  dst->reserve((end_it - begin_it) + log->events.size());
  std::merge(begin_it, end_it, log->events.begin(), log->events.end(),
             std::back_inserter(*dst), [](const Event *l, const Event *r) { return *l < *r; });

  std::unique_lock events_lock(mutex_);
  events_changed_ = true;
  for (auto cam_type : ALL_CAMERAS) {
    encoderIdx_[cam_type].merge(log->encoderIdx[cam_type]);
  }
  events_ = dst;
  // qInfo() << "merge array " << millis_since_boot() - t1 << "size " << events_->size();
}

// class Segment

Segment::Segment(int seg_num, const SegmentFile &file, QObject *parent) : seg_num(seg_num), QObject(parent) {
  // fallback to qlog if rlog not exists.
  const QString &log_file = file.rlog.isEmpty() ? file.qlog : file.rlog;
  if (log_file.isEmpty()) {
    qDebug() << "no log file in segment " << seg_num;
    return;
  }

  loading_ = 1;
  log = new LogReader(log_file);
  QObject::connect(log, &LogReader::finished, [&](bool success) {
    if (--loading_ == 0) emit loaded();
  });

  // start framereader threads
  // fallback to qcamera if camera not exists.
  std::pair<CameraType, QString> cam_files[] = {{RoadCam, file.camera.isEmpty() ? file.qcamera : file.camera},
                                                {DriverCam, file.dcamera},
                                                {WideRoadCam, file.wcamera}};
  for (const auto &[cam_type, file] : cam_files) {
    if (!file.isEmpty()) {
      loading_ += 1;
      FrameReader *fr = frames[cam_type] = new FrameReader(file.toStdString());
      QObject::connect(fr, &FrameReader::finished, [=]() { if(--loading_ == 0) emit loaded(); });
    }
  }
}

Segment::~Segment() {
  qDebug() << QString("remove segment %1").arg(seg_num);
  delete log;
  for (auto f : frames) delete f;
}

// class CameraServer

CameraServer::CameraServer() {
  device_id_ = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  context_ = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id_, NULL, NULL, &err));
}

CameraServer::~CameraServer() {
  delete vipc_server_;
  CL_CHECK(clReleaseContext(context_));
}

void CameraServer::ensureServerForSegment(Segment *seg) {
  static VisionStreamType stream_types[] = {
      [RoadCam] = VISION_STREAM_RGB_BACK,
      [DriverCam] = VISION_STREAM_RGB_FRONT,
      [WideRoadCam] = VISION_STREAM_RGB_WIDE,
  };

  if (vipc_server_) {
    // restart vipc server if frame changed. such as switched between qcameras and cameras.
    for (auto cam_type : ALL_CAMERAS) {
      const FrameReader *fr = seg->frames[cam_type];
      const CameraState *s = camera_states_[cam_type];
      bool frame_changed = false;
      if (fr && fr->valid()) {
        frame_changed = !s || s->width != fr->width || s->height != fr->height;
      } else if (s) {
        frame_changed = true;
      }
      if (frame_changed) {
        qDebug() << "restart vipc server";
        stop();
        break;
      }
    }
  }
  if (!vipc_server_) {
    for (auto cam_type : ALL_CAMERAS) {
      const FrameReader *fr = seg->frames[cam_type];
      if (!fr || !fr->valid()) continue;

      if (!vipc_server_) {
        vipc_server_ = new VisionIpcServer("camerad", device_id_, context_);
      }
      vipc_server_->create_buffers(stream_types[cam_type], UI_BUF_COUNT, true, fr->width, fr->height);

      CameraState *state = new CameraState;
      state->width = fr->width;
      state->height = fr->height;
      state->stream_type = stream_types[cam_type];
      state->thread = std::thread(&CameraServer::cameraThread, this, cam_type, state);
      camera_states_[cam_type] = state;
    }
    if (vipc_server_) {
      vipc_server_->start_listener();
    }
  }
}

void CameraServer::stop() {
  if (!vipc_server_) return;

  // stop camera threads
  exit_ = true;
  for (int i = 0; i < std::size(camera_states_); ++i) {
    if (CameraState *state = camera_states_[i]) {
      camera_states_[i] = nullptr;
      state->thread.join();
      delete state;
    }
  }
  exit_ = false;

  // stop vipc server
  delete vipc_server_;
  vipc_server_ = nullptr;
}

void CameraServer::cameraThread(CameraType cam_type, CameraServer::CameraState *s) {
  while (!exit_) {
    std::pair<std::shared_ptr<Segment>, uint32_t> frame;
    if (!s->queue.try_pop(frame, 20)) continue;

    auto &[seg, segmentId] = frame;
    FrameReader *frm = seg->frames[cam_type];
    if (frm->width != s->width || frm->height != s->height) {
      // eidx is not in the same segment with different frame size
      continue;
    }

    if (uint8_t *data = frm->get(segmentId)) {
      VisionIpcBufExtra extra = {};
      VisionBuf *buf = vipc_server_->get_buffer(s->stream_type);
      memcpy(buf->addr, data, frm->getRGBSize());
      vipc_server_->send(buf, &extra, false);
    }
  }
  qDebug() << "camera thread " << cam_type << " stopped ";
}
