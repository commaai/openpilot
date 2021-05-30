#include "selfdrive/ui/replay/replay.h"

#include "cereal/services.h"
#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/hardware/hw.h"

const int SEGMENT_LENGTH = 60;  // 60s
const int FORWARD_SEGS = 2;
const int BACKWARD_SEGS = 2;

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

  device_id_ = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  context_ = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id_, NULL, NULL, &err));
}

Replay::~Replay() {
  clear();
  delete pm_;
  delete vipc_server_;
  CL_CHECK(clReleaseContext(context_));
}

void Replay::clear() {
  std::unique_lock lk(segment_lock_);

  segments_.clear();
  for (int i = 0; i < std::size(frame_queues_); ++i) {
    delete frame_queues_[i];
    frame_queues_[i] = nullptr;
  }

  current_ts_ = 0;
  seek_ts_ = 0;
  current_segment_ = seek_ts_ = 0;
  road_cam_width_ = road_cam_height_ = 0;
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
  segments_.resize(route_.segments().size());
  qInfo() << "replay route " << route_.name() << ", total segments:" << segments_.size();

  // start threads
  typedef void (Replay::*threadFunc)();
  threadFunc thread_func[] = {&Replay::segmentQueueThread, &Replay::streamThread};
  for (int i = 0; i < std::size(thread_func); ++i) {
    QThread *t = QThread::create(thread_func[i], this);
    connect(t, &QThread::finished, t, &QThread::deleteLater);
    t->start();
  }
  return true;
}

// return nullptr if segment is not loaded
std::shared_ptr<Segment> Replay::getSegment(int n) {
  std::unique_lock lk(segment_lock_);
  auto &seg = segments_[n];
  return (seg != nullptr && seg->loaded) ? seg : nullptr;
}

void Replay::ensureVipcServer(const Segment *seg) {
  static VisionStreamType stream_types[] = {
      [RoadCam] = VISION_STREAM_RGB_BACK,
      [DriverCam] = VISION_STREAM_RGB_FRONT,
      [WideRoadCam] = VISION_STREAM_RGB_WIDE,
  };

  if (vipc_server_) {
    const FrameReader *fr = seg->frames[RoadCam];
    if (fr && fr->valid() && (fr->width != road_cam_width_ || fr->height != road_cam_height_)) {
      // restart vipc server if road camera size changed.(switch between qcameras and cameras)
      delete vipc_server_;
      vipc_server_ = nullptr;
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
      if (cam_type == RoadCam) {
        road_cam_width_ = fr->width;
        road_cam_height_ = fr->height;
      }

      frame_queues_[cam_type] = new SafeQueue<const EncodeIdx *>();
      QThread *t = QThread::create(&Replay::cameraThread, this, cam_type, stream_types[cam_type]);
      connect(t, &QThread::finished, t, &QThread::deleteLater);
      t->start();
    }
    if (vipc_server_) {
      vipc_server_->start_listener();
      qDebug() << "start vipc server";
    }
  }
}

void Replay::seekTo(int to_ts) {
  std::unique_lock lk(segment_lock_);
  if (!route_.segments().size()) return;

  seek_ts_ = to_ts;
  current_segment_ = std::clamp(to_ts / SEGMENT_LENGTH, 0, route_.segments().size() - 1);;
  qInfo() << "seeking to " << seek_ts_;
}

void Replay::relativeSeek(int ts) {
  seekTo(current_ts_ + ts);
}

void Replay::pushFrame(CameraType cam_type, int seg_id, uint32_t frame_id) {
  // do nothing if no video stream for this type
  if (!frame_queues_[cam_type]) return;

  // find encodeIdx in adjacent segments_.
  const EncodeIdx *eidx = nullptr;
  int search_in[] = {seg_id, seg_id - 1, seg_id + 1};
  for (auto idx : search_in) {
    auto seg = getSegment(idx);
    if (seg && (eidx = seg->log->getFrameEncodeIdx(cam_type, frame_id))) {
      frame_queues_[cam_type]->push(eidx);
      return;
    }
  }
  qDebug() << "failed to find eidx for frame " << frame_id << " in segment " << seg_id;
}

// threads

void Replay::segmentQueueThread() {
  // maintain the segment window
  while (!exit_) {
    for (int i = 0; i < segments_.size(); ++i) {
      const int start_idx = std::max(current_segment_ - BACKWARD_SEGS, 0);
      const int end_idx = std::min(current_segment_ + FORWARD_SEGS, (int)segments_.size() -1);
      std::unique_lock lk(segment_lock_);
      if (i >= start_idx && i <= end_idx) {
        // add segment
        if (!segments_[i]) {
          segments_[i] = std::make_shared<Segment>(i, route_.segments()[i]);
        }
      } else {
        // remove segment
        segments_[i].reset();
      }
    }
    QThread::msleep(100);
  }
}

void Replay::cameraThread(CameraType cam_type, VisionStreamType stream_type) {
  while (!exit_) {
    const EncodeIdx *eidx = nullptr;
    if (!frame_queues_[cam_type]->try_pop(eidx, 50)) continue;

    std::shared_ptr<Segment> seg = getSegment(eidx->segmentNum);
    if (!seg) continue;

    FrameReader *frm = seg->frames[cam_type];
    if (uint8_t *data = frm->get(eidx->segmentId)) {
      VisionIpcBufExtra extra = {};
      VisionBuf *buf = vipc_server_->get_buffer(stream_type);
      memcpy(buf->addr, data, frm->getRGBSize());
      vipc_server_->send(buf, &extra, false);
    }
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

    ensureVipcServer(seg.get());
    const Events &events = seg->log->events();

    // TODO: use initData's logMonoTime
    if (route_start_ts == 0) {
      route_start_ts = events.firstKey();
    }

    uint64_t t0 = route_start_ts + (seek_ts_ * 1e9);
    qDebug() << "unlogging at" << (t0 - route_start_ts) / 1e9;

    auto eit = events.lowerBound(t0);
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
          qInfo() << "at " << last_print << "| segment:" << seg->id;
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
          pushFrame(RoadCam, seg->id, e.getRoadCameraState().getFrameId());
        } else if (e.which() == cereal::Event::DRIVER_CAMERA_STATE) {
          pushFrame(DriverCam, seg->id, e.getDriverCameraState().getFrameId());
        } else if (e.which() == cereal::Event::WIDE_ROAD_CAMERA_STATE) {
          pushFrame(WideRoadCam, seg->id, e.getWideRoadCameraState().getFrameId());
        }

        // publish msg
        if (sm_ == nullptr) {
          auto bytes = (*eit)->bytes();
          pm_->send(type.c_str(), (capnp::byte *)bytes.begin(), bytes.size());
        } else {
          // TODO: subMaster is not thread safe. are we sure we need to do this?
          sm_->update_msgs(nanos_since_boot(), {{type, e}});
        }
      }

      ++eit;
    }

    if (current_seek_ts_ == seek_ts_ && eit == events.end()) {
      // move to the next segment
      current_segment_ += 1;
      qDebug() << "move to next segment " << current_segment_;
      seek_ts_ = current_ts_.load();
    }
  }
}

// class Segment

Segment::Segment(int segment_id, const SegmentFiles &files) : id(segment_id) {
  // fallback to qlog if rlog not exists.
  const QString &log_file = files.rlog.isEmpty() ? files.qlog : files.rlog;
  if (log_file.isEmpty()) {
    qDebug() << "no log file in segment " << id;
    return;
  }

  loading = 1;
  log = new LogReader(log_file);
  QObject::connect(log, &LogReader::finished, [=](bool success) {
    --loading;
    loaded = loading == 0;
  });

  // start framereader threads
  auto read_cam_frames = [=](CameraType type, const QString &file) {
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
  read_cam_frames(RoadCam, camera);
  read_cam_frames(DriverCam, files.dcamera);
  read_cam_frames(WideRoadCam, files.wcamera);
}

Segment::~Segment() {
  qDebug() << QString("remove segment %1").arg(id);
  delete log;
  for (auto f : frames) delete f;
}
