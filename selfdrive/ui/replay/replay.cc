#include "selfdrive/ui/replay/replay.h"

#include <capnp/dynamic.h>

#include "cereal/services.h"
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
  qDebug() << "services" << s;

  if (sm_ == nullptr) {
    pm_ = new PubMaster(s);
  }
  events_ = std::make_unique<std::vector<Event *>>();
}

Replay::~Replay() {
  stop();
  delete pm_;
}

bool Replay::start(const QString &routeName) {
  Route route(routeName);
  if (!route.load()) {
    qInfo() << "failed to retrieve files for route" << routeName;
    return false;
  }
  return start(route);
}

bool Replay::start(const Route &route) {
  assert(!running());
  if (!route.segments().size()) return false;

  route_ = route;
  current_segment_ = route_.segments().firstKey();
  qDebug() << "replay route" << route_.name() << "from" << current_segment_ << ", total segments:" << route.segments().size();
  queue_thread_ = std::thread(&Replay::queueSegmentThread, this);
  stream_thread_ = std::thread(&Replay::streamThread, this);
  return true;
}

void Replay::stop() {
  if (!running()) return;

  // wait until threads finished
  camera_server_.stop();
  exit_ = true;
  cv_.notify_one();
  queue_thread_.join();
  stream_thread_.join();

  // clear all
  events_->clear();
  segments_.clear();
  current_ts_ = route_start_ts_ = seek_ts_ = 0;
  current_segment_ = 0;
  exit_ = false;
}

QString Replay::elapsedTime(uint64_t ns) {
  QTime time(0, 0, 0);
  auto a = time.addSecs((ns - route_start_ts_) / 1e9);
  return a.toString("hh:mm:ss");
}

void Replay::seek(int seconds) {
  if (route_start_ts_ > 0) {
    seekTo(route_start_ts_ + seconds * 1e9);
  }
}

void Replay::relativeSeek(int seconds) {
  if (current_ts_ > 0) {
    seekTo(current_ts_ + seconds * 1e9);
  }
}

void Replay::seekTo(uint64_t to_ts) {
  const int segment = (to_ts - route_start_ts_) / 1e9 / SEGMENT_LENGTH;
  if (!route_.segments().contains(segment)) {
    qInfo() << "can't seek to" << elapsedTime(to_ts) << ": segment" << segment << "does not exist.";
    return;
  }

  loading_events_ = true;
  {
    std::unique_lock lk(events_mutex_);
    seek_ts_ = to_ts;
    current_segment_ = segment;
    loading_events_ = false;
  }
  cv_.notify_one();
  qInfo() << "seeking to" << elapsedTime(to_ts);
}

const std::string &Replay::eventSocketName(const Event *e) {
  auto it = eventNameMap.find(e->which);
  if (it == eventNameMap.end()) {
    std::string type;
    KJ_IF_MAYBE(e_, static_cast<capnp::DynamicStruct::Reader>(e->event).which()) {
      type = e_->getProto().getName();
    }
    it = eventNameMap.insert(it, {e->which, socks_.find(type) != socks_.end() ? type : ""});
  }
  return it->second;
}

void Replay::mergeEvents() {
  Segment *seg = qobject_cast<Segment *>(QObject::sender());
  const LogReader *log = seg->log;

  if (route_start_ts_ == 0) {
    route_start_ts_ = log->route_start_ts;
  }

  auto [min_seg, max_seg] = queueSegmentRange();
  uint64_t min_tm = route_start_ts_ + min_seg * SEGMENT_LENGTH * 1e9;
  uint64_t max_tm = route_start_ts_ + (max_seg + 1) * SEGMENT_LENGTH * 1e9;

  auto begin_merge_it = std::lower_bound(events_->begin(), events_->end(), min_tm, [](const Event *e, uint64_t v) {
    return e->mono_time < v;
  });
  if (begin_merge_it == events_->end()) {
    begin_merge_it = events_->begin();
  }
  auto end_merge_it = std::upper_bound(begin_merge_it, events_->end(), max_tm, [](uint64_t v, const Event *e) {
    return v < e->mono_time;
  });

  // merge segment
  std::vector<Event *> *dst = new std::vector<Event *>;
  dst->reserve((end_merge_it - begin_merge_it) + log->events.size());
  std::merge(begin_merge_it, end_merge_it, log->events.begin(), log->events.end(),
             std::back_inserter(*dst), [](const Event *l, const Event *r) { return *l < *r; });

  loading_events_ = true;
  {
    std::unique_lock events_lock(events_mutex_);
    events_.reset(dst);
    seg->loaded = true;
    loading_events_ = false;
  }
  cv_.notify_one();

  // erase segments
  auto it = segments_.begin();
  while (it != segments_.end()) {
    auto &seg = it->second;
    auto &e = seg->log->events;
    if (seg->loaded && (e.back()->mono_time < min_tm || e.front()->mono_time > max_tm)) {
      qDebug() << "erase segment " << it->first;
      it = segments_.erase(it);
    } else {
      ++it;
    }
  }
}

std::pair<int, int> Replay::queueSegmentRange() {
  const auto &rs = route_.segments();
  int cur_idx = std::distance(rs.begin(), rs.lowerBound(current_segment_));
  int i = 0, min = rs.firstKey(), max = rs.lastKey();
  for (auto it = rs.begin(); it != rs.end(); ++it, ++i) {
    if (i <= cur_idx - BACKWARD_SEGS) {
      min = it.key();
    } else if (i >= cur_idx + FORWARD_SEGS) {
      max = it.key();
      break;
    }
  }
  return {min, max};
}

// maintain the segment window
void Replay::queueSegmentThread() {
  while (!exit_) {
    auto [min, max] = queueSegmentRange();
    for (auto it = route_.segments().lowerBound(min); it != route_.segments().upperBound(max); ++it) {
      int n = it.key();
      std::unique_lock lk(segment_mutex_);
      if (segments_.find(n) == segments_.end()) {
        segments_[n] = std::make_unique<Segment>(n, it.value());
        connect(segments_[n].get(), &Segment::finishedRead, this, &Replay::mergeEvents);
      }
    }
    QThread::msleep(50);
  }
}

// return nullptr if segment is not loaded
const Segment *Replay::getSegment(int segment) {
  std::unique_lock lk(segment_mutex_);
  auto it = segments_.find(segment);
  return (it != segments_.end() && it->second->loaded) ? it->second.get() : nullptr;
}

void Replay::pushFrame(int cur_seg_num, CameraType cam_type, uint32_t frame_id) {
  // search encodeIdx in adjacent segments.
  for (int n : {cur_seg_num, cur_seg_num - 1, cur_seg_num + 1}) {
    if (auto seg = getSegment(n)) {
      auto eidxMap = seg->log->encoderIdx[cam_type];
      if (auto eidx = eidxMap.find(frame_id); eidx != eidxMap.end()) {
        camera_server_.pushFrame(cam_type, seg->frames[cam_type], eidx->second.segmentId);
        break;
      }
    }
  }
}

void Replay::streamThread() {
  uint64_t last_print_ts = 0, evt_start_ts = 0;
  cereal::Event::Which current_which = cereal::Event::INIT_DATA;
  while (!exit_) {
    std::unique_lock lk(events_mutex_);
    cv_.wait(lk, [=] { return exit_ || !loading_events_; });
    if (exit_) break;

    auto eit = events_->end();
    uint64_t seek_to = seek_ts_ ? seek_ts_ : current_ts_.load();
    cereal::Event::Which which = seek_ts_ ? cereal::Event::INIT_DATA : current_which;
    // make sure current segment is loaded
    if (auto seg = getSegment(current_segment_)) {
      camera_server_.ensure(seg->frames);
      eit = std::lower_bound(events_->begin(), events_->end(), seek_to, [&](const Event *e, uint64_t v) {
        return e->mono_time < v || (e->mono_time == v && e->which < which);
      });
    }
    if (eit == events_->end()) {
      qDebug() << "waiting for events";
      lk.unlock();
      QThread::msleep(100);
      continue;
    }
    evt_start_ts = (*eit)->mono_time;
    if (seek_to > 0) {
      // do not send the previous event again
      ++eit;
      seek_ts_ = 0;
    }
    uint64_t loop_start_ts = nanos_since_boot();
    while (!exit_ && !loading_events_ && eit != events_->end()) {
      const Event *e = (*eit);
      const std::string &sock_name = eventSocketName(e);
      if (!sock_name.empty()) {
        current_which = e->which;
        current_ts_ = e->mono_time;
        current_segment_ = (e->mono_time - route_start_ts_) / 1e9 / SEGMENT_LENGTH;

        if ((e->mono_time - last_print_ts) > 5 * 1e9) {
          last_print_ts = e->mono_time;
          qInfo().noquote() << "at" << elapsedTime(last_print_ts);
        }

        // keep time
        uint64_t etime = e->mono_time - evt_start_ts;
        uint64_t rtime = nanos_since_boot() - loop_start_ts;
        uint64_t us_behind = ((etime - rtime) * 1e-3) + 0.5;
        if (us_behind > 0 && us_behind < 1e6) {
          QThread::usleep(us_behind);
          //qDebug() << "sleeping" << us_behind << etime << timer.nsecsElapsed();
        }

        // publish frames
        switch (current_which) {
          case cereal::Event::ROAD_CAMERA_STATE:
            pushFrame(current_segment_, RoadCam, e->event.getRoadCameraState().getFrameId());
            break;
          case cereal::Event::DRIVER_CAMERA_STATE:
            pushFrame(current_segment_, DriverCam, e->event.getDriverCameraState().getFrameId());
            break;
          case cereal::Event::WIDE_ROAD_CAMERA_STATE:
            pushFrame(current_segment_, WideRoadCam, e->event.getWideRoadCameraState().getFrameId());
            break;
          default:
            break;
        }

        // publish msg
        if (sm_ == nullptr) {
          auto bytes = e->bytes();
          pm_->send(sock_name.c_str(), (capnp::byte *)bytes.begin(), bytes.size());
        } else {
          // TODO: subMaster is not thread safe.
          sm_->update_msgs(nanos_since_boot(), {{sock_name, e->event}});
        }
      }
      ++eit;
    }
  }
}

// class Segment

Segment::Segment(int seg_num, const SegmentFile &file, QObject *parent) : seg_num(seg_num), QObject(parent) {
  // fallback to qlog if rlog not exists.
  const QString &log_file = file.rlog.isEmpty() ? file.qlog : file.rlog;
  if (log_file.isEmpty()) {
    qDebug() << "no log file in segment" << seg_num;
    return;
  }

  loading_ = 1;
  log = new LogReader(log_file, this);
  QObject::connect(log, &LogReader::finished, [&](bool success) {
    if (--loading_ == 0) emit finishedRead();
  });

  // fallback to qcamera if camera not exists.
  std::pair<CameraType, QString> cam_files[] = {{RoadCam, file.camera.isEmpty() ? file.qcamera : file.camera},
                                                {DriverCam, file.dcamera},
                                                {WideRoadCam, file.wcamera}};
  for (const auto &[cam_type, file] : cam_files) {
    if (!file.isEmpty()) {
      loading_ += 1;
      frames[cam_type] = std::make_shared<FrameReader>(file.toStdString(), this);
      QObject::connect(frames[cam_type].get(), &FrameReader::finished, [=]() { if(--loading_ == 0) emit finishedRead(); });
    }
  }
}
