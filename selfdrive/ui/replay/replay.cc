#include "selfdrive/ui/replay/replay.h"

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
  // start threads
  threads_.push_back(std::thread(&Replay::segmentQueueThread, this));
  threads_.push_back(std::thread(&Replay::streamThread, this));
}

Replay::~Replay() {
  exit_ = true;
  for (auto &t : threads_) {
    t.join();
  }
  clear();
  delete pm_;
}

bool Replay::load(const QString &routeName) {
  Route route(routeName);
  if (!route.load()) {
    qInfo() << "failed to retrieve files for route " << routeName;
    return false;
  }
  return load(route);
}

bool Replay::load(const Route &route) {
  if (!route.segments().size()) return false;

  clear();
  std::unique_lock lk(segment_lock_);
  route_ = route;
  current_segment_ = route_.segments().firstKey();
  qInfo() << "replay route " << route_.name() << " from " << current_segment_ << ", total segments:" << route.segments().size();

  return true;
}


void Replay::clear() {
  std::unique_lock lk(segment_lock_);

  camera_server_.stop();
  segments_.clear();
  current_ts_ = seek_ts_ = 0;
  current_segment_ = 0;
}

void Replay::relativeSeek(int ts) {
  seekTo(current_ts_ + ts);
}

void Replay::seekTo(int to_ts) {
  std::unique_lock lk(segment_lock_);
  if (!route_.segments().size()) return;

  int seg_num = std::clamp(to_ts / SEGMENT_LENGTH, 0, route_.maxSegmentNum());
  // skip to next or prev segment if seg_num is missing
  if (!route_.segments().contains(seg_num)) {
    int n = seg_num > current_segment_ ? route_.nextSegNum(seg_num) : route_.prevSegNum(seg_num);
    if (n != -1) {
      seg_num = n;
    }
  }
  seek_ts_ = to_ts;
  current_segment_ = seg_num;
  qInfo() << "seeking to " << seek_ts_;
}

void Replay::pushFrame(CameraType cam_type, int seg_num, uint32_t frame_id) {
  if (!camera_server_.hasCamera(cam_type)) return;

  // find encodeIdx in adjacent segments_.
  const EncodeIdx *eidx = nullptr;
  int search_in[] = {seg_num, seg_num - 1, seg_num + 1};
  for (auto n : search_in) {
    auto seg = getSegment(n);
    if (seg && (eidx = seg->log->getFrameEncodeIdx(cam_type, frame_id))) {
      camera_server_.pushFrame(cam_type, seg, eidx->segmentId);
      return;
    }
  }
  qDebug() << "failed to find eidx for frame " << frame_id << " in segment " << seg_num;
}

// return nullptr if segment is not loaded
std::shared_ptr<Segment> Replay::getSegment(int n) {
  std::unique_lock lk(segment_lock_);
  auto it = segments_.find(n);
  return (it != segments_.end() && it->second->loaded) ? it->second : nullptr;
}

void Replay::segmentQueueThread() {
  // maintain the segment window
  while (!exit_) {
    QList<int> seg_nums = route_.segments().keys();
    int idx = std::max(seg_nums.indexOf(current_segment_), 0);
    for (int i = 0; i < seg_nums.size(); ++i) {
      const int seg_num = seg_nums[i];
      const int start_idx = std::max(idx - BACKWARD_SEGS, 0);
      const int end_idx = std::min(idx + FORWARD_SEGS, (int)seg_nums.size() - 1);
      std::unique_lock lk(segment_lock_);
      if (i >= start_idx && i <= end_idx) {
        // add segment
        if (segments_.find(seg_num) == segments_.end()) {
          segments_[seg_num] = std::make_shared<Segment>(i, route_.segments()[seg_num]);
        }
      } else {
        // remove segment
        segments_.erase(seg_num);
      }
    }
    QThread::msleep(100);
  }
}

void Replay::streamThread() {
  QElapsedTimer timer;
  timer.start();

  uint64_t route_start_ts = 0;
  int64_t last_print = 0;
  while (!exit_) {
    std::shared_ptr<Segment> seg = getSegment(current_segment_);
    if (!seg) {
      qDebug() << "waiting for events";
      QThread::msleep(100);
      continue;
    }
    camera_server_.ensureServerForSegment(seg.get());

    const Events &events = seg->log->events();
    // TODO: use initData's logMonoTime
    if (route_start_ts == 0) {
      route_start_ts = events.firstKey();
    }

    uint64_t t0 = route_start_ts + (seek_ts_ * 1e9);
    auto eit = events.lowerBound(t0);
    if (eit != events.end()) {
      // adjust t0 to current event's tm.
      t0 = eit.key();
      seek_ts_ = (t0 - route_start_ts) / 1e9;
    }
    qDebug() << "unlogging at" << seek_ts_;
    uint64_t t0r = timer.nsecsElapsed();
    int current_seek_ts_ = seek_ts_;
    while (!exit_ && current_seek_ts_ == seek_ts_ && eit != events.end()) {
      cereal::Event::Reader e = (*eit)->event();
      std::string type;
      KJ_IF_MAYBE(e_, static_cast<capnp::DynamicStruct::Reader>(e).which()) {
        type = e_->getProto().getName();
      }

      uint64_t tm = e.getLogMonoTime();
      current_ts_ = std::max(tm - route_start_ts, (unsigned long)0) / 1e9;

      if (socks_.find(type) != socks_.end()) {
        if (std::abs(current_ts_ - last_print) > 5.0) {
          last_print = current_ts_;
          qInfo() << "at " << last_print << "| segment:" << seg->seg_num;
        }

        // keep time
        long etime = tm - t0;
        long rtime = timer.nsecsElapsed() - t0r;
        long us_behind = ((etime - rtime) * 1e-3) + 0.5;
        if (us_behind > 0 && us_behind < 1e6) {
          QThread::usleep(us_behind);
          //qDebug() << "sleeping" << us_behind << etime << timer.nsecsElapsed();
        }

        // publish frames
        if (e.which() == cereal::Event::ROAD_CAMERA_STATE) {
          pushFrame(RoadCam, seg->seg_num, e.getRoadCameraState().getFrameId());
        } else if (e.which() == cereal::Event::DRIVER_CAMERA_STATE) {
          pushFrame(DriverCam, seg->seg_num, e.getDriverCameraState().getFrameId());
        } else if (e.which() == cereal::Event::WIDE_ROAD_CAMERA_STATE) {
          pushFrame(WideRoadCam, seg->seg_num, e.getWideRoadCameraState().getFrameId());
        }

        // publish msg
        if (sm_ == nullptr) {
          auto bytes = (*eit)->bytes();
          pm_->send(type.c_str(), (capnp::byte *)bytes.begin(), bytes.size());
        } else {
          // TODO: subMaster is not thread safe.
          sm_->update_msgs(nanos_since_boot(), {{type, e}});
        }
      }

      ++eit;
    }

    if (current_seek_ts_ == seek_ts_ && eit == events.end()) {
      // move to the next segment
      seek_ts_ = current_ts_.load();
      if (int n = route_.nextSegNum(current_segment_); n != -1) {
        current_segment_ = n;
        qDebug() << "move to next segment " << current_segment_;
      } else {
        qDebug() << "reach the end of segments, stop replay.";
      }
    }
  }
}

// class Segment

Segment::Segment(int seg_num, const SegmentFiles &files) : seg_num(seg_num) {
  // fallback to qlog if rlog not exists.
  const QString &log_file = files.rlog.isEmpty() ? files.qlog : files.rlog;
  if (log_file.isEmpty()) {
    qDebug() << "no log file in segment " << seg_num;
    return;
  }

  loading = 1;
  log = new LogReader(log_file);
  QObject::connect(log, &LogReader::finished, [=](bool success) {
    --loading;
    loaded = loading == 0;
  });

  // start framereader threads
  auto read_frames = [=](CameraType type, const QString &file) {
    if (!file.isEmpty()) {
      loading += 1;
      FrameReader *fr = frames[type] = new FrameReader(file.toStdString());
      QObject::connect(fr, &FrameReader::finished, [=](bool success) {
        --loading;
        loaded = loading == 0;
      });
      fr->start();
    }
  };

  // fallback to qcamera if camera not exists.
  const QString &camera = files.camera.isEmpty() ? files.qcamera : files.camera;
  read_frames(RoadCam, camera);
  read_frames(DriverCam, files.dcamera);
  read_frames(WideRoadCam, files.wcamera);
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
    // restart vipc server if camera changed. such as switched between qcameras and cameras.
    for (auto cam_type : ALL_CAMERAS) {
      const FrameReader *fr = seg->frames[cam_type];
      CameraState *state = camera_states_[cam_type];
      bool camera_changed = false;
      if (fr && fr->valid()) {
        camera_changed = !state || state->width != fr->width || state->height != fr->height; 
      } else if (state != nullptr) {
        camera_changed = true;
      }
      if (camera_changed) {
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
      
      CameraState *state = camera_states_[cam_type] = new CameraState;
      state->width = fr->width;
      state->height = fr->height;
      state->stream_type = stream_types[cam_type];
      state->thread = std::thread(&CameraServer::cameraThread, this, cam_type, state);
    }
    if (vipc_server_) {
      vipc_server_->start_listener();
    }
  }
}

void CameraServer::cameraThread(CameraType cam_type, CameraServer::CameraState *s) {
  while (!exit_) {
    std::pair<std::shared_ptr<Segment>, uint32_t> frame;
    if (!s->queue.try_pop(frame, 20)) continue;

    auto [seg, segmentId] = frame;

    FrameReader *frm = seg->frames[cam_type];
    if (uint8_t *data = frm->get(segmentId)) {
      VisionIpcBufExtra extra = {};
      VisionBuf *buf = vipc_server_->get_buffer(s->stream_type);
      memcpy(buf->addr, data, frm->getRGBSize());
      vipc_server_->send(buf, &extra, false);
    }
  }
  qDebug() << "camera thread " << cam_type << " stopped ";
}

void CameraServer::stop() {
  if (!vipc_server_) return;

  // stop threads
  exit_ = true;
  for (int i = 0; i < std::size(camera_states_); ++i) {
    if (CameraState *state = camera_states_[i]) {
      state->thread.join();
      delete state;
      camera_states_[i] = nullptr;
    }
  }
  exit_ = false;

  delete vipc_server_;
  vipc_server_ = nullptr;
}
