#include "selfdrive/ui/replay/replay.h"

#include <capnp/dynamic.h>

#include "cereal/services.h"
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
  qDebug() << "services" << s;

  if (sm_ == nullptr) {
    pm_ = new PubMaster(s);
  }
  events_ = std::make_unique<std::vector<Event *>>();

  connect(this, &Replay::segmentChanged, this, &Replay::queueSegment);
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
  queueSegment(current_segment_);
  stream_thread_ = std::thread(&Replay::streamThread, this);
  return true;
}

void Replay::stop() {
  if (!running()) return;

  // wait until threads finished
  camera_server_.stop();
  exit_ = true;
  cv_.notify_one();
  stream_thread_.join();

  // clear all
  events_->clear();
  segments_.clear();
  current_ts_ = route_start_ts_ = seek_ts_ = 0;
  current_segment_ = 0;
  exit_ = false;
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
  const int segment = int64_t(to_ts - route_start_ts_) / 1e9 / SEGMENT_LENGTH;
  auto seg = getSegment(segment);
  if ((seg && seg->failed()) || !route_.segments().contains(segment)) {
    qInfo() << "can't seek to" << elapsedTime(to_ts) << ": segment" << segment << "does not exist.";
    return;
  }
  emit segmentChanged(segment);
  qInfo() << "seeking to" << elapsedTime(to_ts);

  loading_events_ = true;
  std::unique_lock lk(events_mutex_);
  seek_ts_ = to_ts;
  current_segment_ = segment;
  loading_events_ = false;
  cv_.notify_one();
}

std::pair<int, int> Replay::cacheSegmentRange(int seg_num) {
  QList<int> s = route_.segments().keys();
  int i = s.indexOf(seg_num);
  return {s[std::max(i - BACKWARD_SEGS, 0)],
          s[std::min(i + FORWARD_SEGS, s.size() - 1)]};
}

void Replay::queueSegment(int seg_num) {
  auto[_, max] = cacheSegmentRange(seg_num);
  for (auto it = route_.segments().begin(); it != route_.segments().end(); ++it) {
    if (it.key() >= seg_num && it.key() <= max) {
      std::unique_lock lk(segment_mutex_);
      if (segments_.find(it.key()) == segments_.end()) {
        auto seg = segments_[it.key()] = new Segment(this);
        connect(seg, &Segment::loadFinished, this, &Replay::mergeEvents);
        seg->load(it.key(), it.value());
      }
    }
  }
}

const Segment *Replay::getSegment(int segment) {
  std::unique_lock lk(segment_mutex_);
  auto it = segments_.find(segment);
  return (it != segments_.end() && it->second->loaded()) ? it->second : nullptr;
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

void Replay::mergeEvents(bool success) {
  if (!success) return;

  Segment *seg = qobject_cast<Segment *>(QObject::sender());

  auto [min_seg, max_seg] = cacheSegmentRange(current_segment_);
  if (seg->seg_num >= min_seg && seg->seg_num <= max_seg) {
    // parse events
    auto insertEidx = [&](CameraType type, const cereal::EncodeIndex::Reader &e) {
      eidx[type][e.getFrameId()] = {e.getSegmentNum(), e.getSegmentId()};
    };

    for (auto evt : seg->log->events) {
      switch (evt->which) {
        case cereal::Event::INIT_DATA:
          route_start_ts_ = evt->mono_time;
          break;
        case cereal::Event::ROAD_ENCODE_IDX:
          insertEidx(RoadCam, evt->event.getRoadEncodeIdx());
          break;
        case cereal::Event::DRIVER_ENCODE_IDX:
          insertEidx(DriverCam, evt->event.getDriverEncodeIdx());
          break;
        case cereal::Event::WIDE_ROAD_ENCODE_IDX:
          insertEidx(WideRoadCam, evt->event.getWideRoadEncodeIdx());
          break;
        default:
          break;
      }
    }

    // merge events
    std::vector<Event *> *dst = new std::vector<Event *>;
    dst->reserve(events_->size() + seg->log->events.size());
    for (auto it = segments_.cbegin(); it != segments_.cend(); ++it) {
      if (it->first >= min_seg && it->first <= max_seg && it->second->loaded()) {
        auto &e = it->second->log->events;
        auto middle = dst->insert(dst->end(), e.begin(), e.end());
        std::inplace_merge(dst->begin(), middle, dst->end(),
                          [](const Event *l, const Event *r) { return *l < *r; });
      }
    }

    // notify stream thread
    loading_events_ = true;
    std::unique_lock events_lock(events_mutex_);
    events_.reset(dst);
    loading_events_ = false;
    cv_.notify_one();
  }

  // remove segments
  std::unique_lock lk(segment_mutex_);
  auto it = segments_.begin();
  while (it != segments_.end()) {
    if (it->second->seg_num < min_seg || it->second->seg_num > max_seg) {
      it = segments_.erase(it);
    } else {
      ++it;
    }
  }
}

void Replay::pushFrame(CameraType cam_type, uint32_t frame_id) {
  if (auto e = eidx[cam_type].find(frame_id); e != eidx[cam_type].end()) {
    if (auto seg = getSegment(e->second.seg_num)) {
      camera_server_.pushFrame(cam_type, seg, e->second.encode_id);
    }
  }
}

void Replay::streamThread() {
  uint64_t last_print_ts = 0, evt_start_ts = 0;
  cereal::Event::Which current_which = cereal::Event::INIT_DATA;
  bool waiting_printed = false;
  while (!exit_) {
    std::unique_lock lk(events_mutex_);
    cv_.wait(lk, [=] { return exit_ || !loading_events_; });
    if (exit_) break;

    uint64_t seek_to = seek_ts_ ? seek_ts_ : current_ts_.load();
    cereal::Event::Which which = seek_ts_ ? cereal::Event::INIT_DATA : current_which;
    auto eit = events_->end();
    // wait until current segment is loaded
    if (getSegment(current_segment_)) {
      eit = std::lower_bound(events_->begin(), events_->end(), seek_to, [&](const Event *e, uint64_t v) {
        return e->mono_time < v || (e->mono_time == v && e->which <= which);
      });
    }
    if (eit == events_->end()) {
      if (!waiting_printed) {
        qInfo() << "waiting for events...";
        waiting_printed = true;
      }
      lk.unlock();
      QThread::msleep(50);
      continue;
    }
    waiting_printed = false;
    seek_ts_ = 0;
    evt_start_ts = (*eit)->mono_time;
    uint64_t loop_start_ts = nanos_since_boot();

    while (!exit_ && !loading_events_ && eit != events_->end()) {
      const Event *e = (*eit);
      const std::string &sock_name = eventSocketName(e);
      if (!sock_name.empty()) {
        current_which = e->which;
        current_ts_ = e->mono_time;
        int seg_num = (e->mono_time - route_start_ts_) / 1e9 / SEGMENT_LENGTH;
        if (current_segment_ != seg_num) {
          current_segment_ = seg_num;
          emit segmentChanged(seg_num);
        }

        if ((e->mono_time - last_print_ts) > 5 * 1e9) {
          last_print_ts = e->mono_time;
          qInfo().noquote() << "at" << elapsedTime(last_print_ts);
        }

        // keep time
        int64_t etime = e->mono_time - evt_start_ts;
        int64_t rtime = nanos_since_boot() - loop_start_ts;
        int64_t us_behind = ((etime - rtime) * 1e-3) + 0.5;
        if (us_behind > 0 && us_behind < 1e6) {
          QThread::usleep(us_behind);
        }

        // publish frames
        switch (current_which) {
          case cereal::Event::ROAD_CAMERA_STATE:
            pushFrame(RoadCam, e->event.getRoadCameraState().getFrameId());
            break;
          case cereal::Event::DRIVER_CAMERA_STATE:
            pushFrame(DriverCam, e->event.getDriverCameraState().getFrameId());
            break;
          case cereal::Event::WIDE_ROAD_CAMERA_STATE:
            pushFrame(WideRoadCam, e->event.getWideRoadCameraState().getFrameId());
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

void Segment::load(int seg_num, const SegmentFile &file) {
  this->seg_num = seg_num;
  // fallback to qlog if rlog not exists.
  const QString &log_file = file.rlog.isEmpty() ? file.qlog : file.rlog;
  if (log_file.isEmpty()) {
    qInfo() << "no log file in segment" << seg_num;
    emit loadFinished(false);
    return;
  }

  loading_ = 1;
  log = new LogReader(log_file);
  QObject::connect(log, &LogReader::finished, [&](bool success) {
    failed_ += !success;
    if (--loading_ == 0) emit loadFinished(failed_ == 0);
  });

  // fallback to qcamera if camera not exists.
  const std::pair<CameraType, QString> cam_files[] = {
      {RoadCam, file.camera.isEmpty() ? file.qcamera : file.camera},
      {DriverCam, file.dcamera},
      {WideRoadCam, file.wcamera}};
  for (const auto &[cam_type, file] : cam_files) {
    if (!file.isEmpty()) {
      loading_ += 1;
      frames[cam_type] = new FrameReader(file.toStdString());
      frame_thread_[cam_type] = std::thread([=,type=cam_type]() {
        bool ret = frames[type]->FrameReader::process();
        failed_ += !ret;
        if (--loading_ == 0) emit loadFinished(failed_ == 0);
      });
    }
  }
  qDebug() << "loading segment" << seg_num;
}

Segment::~Segment() {
  for (auto cam_type : ALL_CAMERAS) {
    if (FrameReader *fr = frames[cam_type]) {
      fr->stop();
      frame_thread_[cam_type].join();
      delete fr;
    }
  }
  delete log;
  qDebug() << "remove segment " << seg_num;
}
