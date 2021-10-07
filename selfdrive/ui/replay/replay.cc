#include "selfdrive/ui/replay/replay.h"

#include <QApplication>
#include <QDebug>

#include <capnp/dynamic.h>
#include "cereal/services.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/replay/util.h"

Replay::Replay(QString route, QStringList allow, QStringList block, SubMaster *sm_, bool dcam, bool ecam, QObject *parent)
    : sm(sm_), load_dcam(dcam), load_ecam(ecam), QObject(parent) {
  std::vector<const char *> s;
  auto event_struct = capnp::Schema::from<cereal::Event>().asStruct();
  sockets_.resize(event_struct.getUnionFields().size());
  for (const auto &it : services) {
    if ((allow.size() == 0 || allow.contains(it.name)) &&
        !block.contains(it.name)) {
      s.push_back(it.name);
      uint16_t which = event_struct.getFieldByName(it.name).getProto().getDiscriminantValue();
      sockets_[which] = it.name;
    }
  }
  qDebug() << "services " << s;

  if (sm == nullptr) {
    pm = new PubMaster(s);
  }
  route_ = std::make_unique<Route>(route);
  events_ = new std::vector<Event *>();
  // doSeek & queueSegment are always executed in the same thread
  connect(this, &Replay::seekTo, this, &Replay::doSeek);
  connect(this, &Replay::segmentChanged, this, &Replay::queueSegment);
}

Replay::~Replay() {
  qDebug() << "shutdown: in progress...";
  exit_ = true;
  updating_events_ = true;
  if (stream_thread_) {
    stream_cv_.notify_one();
    stream_thread_->quit();
    stream_thread_->wait();
  }

  delete pm;
  delete events_;
  segments_.clear();
  camera_server_.reset(nullptr);
  qDebug() << "shutdown: done";
}

bool Replay::load() {
  if (!route_->load()) {
    qDebug() << "failed to load route" << route_->name() << "from server";
    return false;
  }

  for (int i = 0; i < route_->size(); ++i) {
    const SegmentFile &f = route_->at(i);
    if ((!f.rlog.isEmpty() || !f.qlog.isEmpty()) && (!f.road_cam.isEmpty() || !f.qcamera.isEmpty())) {
      segments_[i] = nullptr;
    }
  }
  if (segments_.empty()) {
    qDebug() << "no valid segments in route" << route_->name();
    return false;
  }
  qDebug() << "load route" << route_->name() << "with" << segments_.size() << "valid segments";
  return true;
}

void Replay::start(int seconds) {
  seekTo(seconds, false);

  camera_server_ = std::make_unique<CameraServer>();
  // start stream thread
  stream_thread_ = new QThread(this);
  QObject::connect(stream_thread_, &QThread::started, [=]() { stream(); });
  stream_thread_->start();
}

void Replay::updateEvents(const std::function<bool()> &lambda) {
  // set updating_events to true to force stream thread relase the lock and wait for evnets_udpated.
  updating_events_ = true;
  {
    std::unique_lock lk(stream_lock_);
    events_updated_ = lambda();
    updating_events_ = false;
  }
  stream_cv_.notify_one();
}

void Replay::doSeek(int seconds, bool relative) {
  updateEvents([&]() {
    if (relative) {
      seconds += currentSeconds();
    }
    qInfo() << "seeking to" << seconds;
    cur_mono_time_ = route_start_ts_ + std::clamp(seconds, 0, (int)segments_.rbegin()->first * 60) * 1e9;
    current_segment_ = std::min(seconds / 60, (int)segments_.rbegin()->first - 1);
    return false;
  });
  queueSegment();
}

void Replay::pause(bool pause) {
  updateEvents([=]() {
    qDebug() << (pause ? "paused..." : "resuming");
    if (pause) {
      qInfo() << "at " << currentSeconds() << "s";
    }
    paused_ = pause;
    return true;
  });
}

void Replay::setCurrentSegment(int n) {
  if (current_segment_.exchange(n) != n) {
    emit segmentChanged();
  }
}

// maintain the segment window
void Replay::queueSegment() {
  // forward fetch segments
  SegmentMap::iterator begin, end;
  begin = end = segments_.lower_bound(current_segment_);
  for (int fwd = 0; end != segments_.end() && fwd <= FORWARD_SEGS; ++end, ++fwd) {
    auto &[n, seg] = *end;
    if (!seg) {
      seg = std::make_unique<Segment>(n, route_->at(n), load_dcam, load_ecam);
      QObject::connect(seg.get(), &Segment::loadFinished, this, &Replay::queueSegment);
    }
  }
  // merge segments
  mergeSegments(begin, end);
  // free segments out of current semgnt window.
  for (auto it = segments_.begin(); it != begin; ++it) {
    it->second.reset(nullptr);
  }
  for (auto it = end; it != segments_.end(); ++it) {
    it->second.reset(nullptr);
  }
}

void Replay::mergeSegments(const SegmentMap::iterator &begin, const SegmentMap::iterator &end) {
  // segments must be merged in sequence.
  std::vector<int> segments_need_merge;
  for (auto it = begin; it != end && it->second->isLoaded(); ++it) {
    segments_need_merge.push_back(it->first);
  }

  if (segments_need_merge != segments_merged_) {
    qDebug() << "merge segments" << segments_need_merge;
    // merge & sort events
    std::vector<Event *> *new_events = new std::vector<Event *>();
    new_events->reserve(std::accumulate(segments_need_merge.begin(), segments_need_merge.end(), 0,
                                        [=](int v, int n) { return v + segments_[n]->log->events.size(); }));
    for (int n : segments_need_merge) {
      auto &log = segments_[n]->log;
      auto middle = new_events->insert(new_events->end(), log->events.begin(), log->events.end());
      std::inplace_merge(new_events->begin(), middle, new_events->end(), Event::lessThan());
    }
    // update events
    auto prev_events = events_;
    updateEvents([&]() {
      if (route_start_ts_ == 0) {
        // get route start time from initData
        auto it = std::find_if(new_events->begin(), new_events->end(), [=](auto e) { return e->which == cereal::Event::Which::INIT_DATA; });
        if (it != new_events->end()) {
          route_start_ts_ = (*it)->mono_time;
          // cur_mono_time_ is set by seekTo in start() before get route_start_ts_
          cur_mono_time_ += route_start_ts_;
        }
      }

      events_ = new_events;
      segments_merged_ = segments_need_merge;
      return true;
    });
    delete prev_events;
  } else {
    updateEvents([=]() { return begin->second->isLoaded(); });
  }
}

void Replay::publishFrame(const Event *e) {
  auto publish = [=](CameraType cam_type, const cereal::EncodeIndex::Reader &eidx) {
    int segment = eidx.getSegmentNum();
    bool segment_loaded = std::find_if(segments_merged_.begin(), segments_merged_.end(), [=](int n) { return n == segment; }) != segments_merged_.end();
    if (segment_loaded && segments_[segment]->frames[cam_type] && eidx.getType() == cereal::EncodeIndex::Type::FULL_H_E_V_C) {
      camera_server_->pushFrame(cam_type, segments_[segment]->frames[cam_type].get(), eidx);
    }
  };
  if (e->which == cereal::Event::ROAD_ENCODE_IDX) {
    publish(RoadCam, e->event.getRoadEncodeIdx());
  } else if (e->which == cereal::Event::DRIVER_ENCODE_IDX) {
    publish(DriverCam, e->event.getDriverEncodeIdx());
  } else if (e->which == cereal::Event::WIDE_ROAD_ENCODE_IDX) {
    publish(WideRoadCam, e->event.getWideRoadEncodeIdx());
  }
}

void Replay::stream() {
  float last_print = 0;
  cereal::Event::Which cur_which = cereal::Event::Which::INIT_DATA;

  std::unique_lock lk(stream_lock_);

  while (true) {
    stream_cv_.wait(lk, [=]() { return exit_ || (events_updated_ && !paused_); });
    events_updated_ = false;
    if (exit_) break;

    Event cur_event(cur_which, cur_mono_time_);
    auto eit = std::upper_bound(events_->begin(), events_->end(), &cur_event, Event::lessThan());
    if (eit == events_->end()) {
      qDebug() << "waiting for events...";
      continue;
    }

    uint64_t evt_start_ts = cur_mono_time_;
    uint64_t loop_start_ts = nanos_since_boot();

    for (auto end = events_->end(); !updating_events_ && eit != end; ++eit) {
      const Event *evt = (*eit);
      cur_which = evt->which;
      cur_mono_time_ = evt->mono_time;
      int current_ts = currentSeconds();
      if (last_print > current_ts || (current_ts - last_print) > 5.0) {
        last_print = current_ts;
        qInfo() << "at " << current_ts << "s";
      }
      setCurrentSegment(current_ts / 60);

      if (cur_which < sockets_.size() && sockets_[cur_which] != nullptr) {
        // keep time
        long etime = cur_mono_time_ - evt_start_ts;
        long rtime = nanos_since_boot() - loop_start_ts;
        long behind_ns = etime - rtime;
        if (behind_ns > 0) {
          precise_nano_sleep(behind_ns);
        }

        if (evt->frame) {
          publishFrame(evt);
        } else {
          // publish msg
          if (sm == nullptr) {
            auto bytes = evt->bytes();
            if (-1 == pm->send(sockets_[cur_which], (capnp::byte *)bytes.begin(), bytes.size())) {
              qDebug() << "stop publish" << sockets_[cur_which] << "due to multiple publishers error";
              sockets_[cur_which] = nullptr;
            }
          } else {
            sm->update_msgs(nanos_since_boot(), {{sockets_[cur_which], evt->event}});
          }
        }
      }
    }

    // wait for frame to be sent before unlock.(frameReader may be deleted after unlock)
    camera_server_->waitFinish();
  }
}
