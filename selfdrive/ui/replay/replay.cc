#include "selfdrive/ui/replay/replay.h"

#include <QApplication>
#include <QDebug>

#include <capnp/dynamic.h>
#include "cereal/services.h"
#include "selfdrive/common/params.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/replay/util.h"

Replay::Replay(QString route, QStringList allow, QStringList block, SubMaster *sm_, uint32_t flags, QString data_dir, QObject *parent)
    : sm(sm_), flags_(flags), QObject(parent) {
  std::vector<const char *> s;
  auto event_struct = capnp::Schema::from<cereal::Event>().asStruct();
  sockets_.resize(event_struct.getUnionFields().size());
  for (const auto &it : services) {
    if ((allow.empty() || allow.contains(it.name)) && !block.contains(it.name)) {
      uint16_t which = event_struct.getFieldByName(it.name).getProto().getDiscriminantValue();
      sockets_[which] = it.name;
      s.push_back(it.name);
    }
  }
  qDebug() << "services " << s;

  if (sm == nullptr) {
    pm = std::make_unique<PubMaster>(s);
  }
  route_ = std::make_unique<Route>(route, data_dir);
  events_ = std::make_unique<std::vector<Event *>>();
  new_events_ = std::make_unique<std::vector<Event *>>();

  qRegisterMetaType<FindFlag>("FindFlag");
  connect(this, &Replay::seekTo, this, &Replay::doSeek);
  connect(this, &Replay::seekToFlag, this, &Replay::doSeekToFlag);
  connect(this, &Replay::segmentChanged, this, &Replay::queueSegment);
}

Replay::~Replay() {
  stop();
}

void Replay::stop() {
  if (!stream_thread_ && segments_.empty()) return;

  qDebug() << "shutdown: in progress...";
  if (stream_thread_ != nullptr) {
    exit_ = updating_events_ = true;
    stream_cv_.notify_one();
    stream_thread_->quit();
    stream_thread_->wait();
    stream_thread_ = nullptr;
  }
  segments_.clear();
  camera_server_.reset(nullptr);
  qDebug() << "shutdown: done";
}

bool Replay::load() {
  if (!route_->load()) {
    qCritical() << "failed to load route" << route_->name() << "from server";
    return false;
  }

  for (auto &[n, f] : route_->segments()) {
    if ((!f.rlog.isEmpty() || !f.qlog.isEmpty()) && (!f.road_cam.isEmpty() || !f.qcamera.isEmpty())) {
      segments_[n] = nullptr;
    }
  }
  if (segments_.empty()) {
    qCritical() << "no valid segments in route" << route_->name();
    return false;
  }
  qInfo() << "load route" << route_->name() << "with" << segments_.size() << "valid segments";
  return true;
}

void Replay::start(int seconds) {
  seekTo(route_->identifier().segment_id * 60 + seconds, false);
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
  if (segments_.empty()) return;

  updateEvents([&]() {
    if (relative) {
      seconds += currentSeconds();
    }
    seconds = std::max(0, seconds);
    int seg = seconds / 60;
    if (segments_.find(seg) == segments_.end()) {
      qWarning() << "can't seek to" << seconds << "s, segment" << seg << "is invalid";
      return true;
    }

    qInfo() << "seeking to" << seconds << "s, segment" << seg;
    current_segment_ = seg;
    cur_mono_time_ = route_start_ts_ + seconds * 1e9;
    return isSegmentMerged(seg);
  });
  queueSegment();
}

void Replay::doSeekToFlag(FindFlag flag) {
  if (flag == FindFlag::nextEngagement) {
    qInfo() << "seeking to the next engagement...";  
  } else {
    qInfo() << "seeking to the disengagement...";  
  }

  updateEvents([&]() {
    auto next = find(flag);
    if (next) {
      cur_mono_time_ = *next;
      current_segment_ = currentSeconds() / 60;
      return isSegmentMerged(current_segment_);
    }
    qWarning() << "seeking failed";
    return true;
  });

  queueSegment();
}

std::optional<uint64_t> Replay::find(FindFlag flag) {
  for (auto &[n, _] : segments_) {
    if (n < current_segment_) continue;

    LogReader log;
    if (!log.load(route_->at(n).qlog.toStdString(), nullptr, true, 0, 3)) continue;

    for (auto evt : log.events) {
      if (evt->mono_time > cur_mono_time_) {
        if (flag == FindFlag::nextEngagement) {
          if (evt->which == cereal::Event::Which::CONTROLS_STATE && evt->event.getControlsState().getEnabled()) {
            return evt->mono_time - 2 * 1e9;
          }
        } else if (flag == FindFlag::nextDisEngagement) {
          if (evt->which == cereal::Event::Which::CONTROLS_STATE && !evt->event.getControlsState().getEnabled()) {
            return evt->mono_time - 2 * 1e9;
          }
        }
      }
    }
  }
  return std::nullopt;
}

void Replay::nextDisengagement() {
}

void Replay::pause(bool pause) {
  updateEvents([=]() {
    qInfo() << (pause ? "paused..." : "resuming");
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

void Replay::segmentLoadFinished(bool success) {
  if (!success) {
    Segment *seg = qobject_cast<Segment *>(sender());
    qWarning() << "failed to load segment " << seg->seg_num << ", removing it from current replay list";
    segments_.erase(seg->seg_num);
  }
  queueSegment();
}

void Replay::queueSegment() {
  if (segments_.empty()) return;

  SegmentMap::iterator cur, end;
  cur = end = segments_.lower_bound(std::min(current_segment_.load(), segments_.rbegin()->first));
  for (int i = 0; end != segments_.end() && i <= FORWARD_SEGS; ++i) {
    ++end;
  }
  // load one segment at a time
  for (auto it = cur; it != end; ++it) {
    if (!it->second) {
      if (it == cur || std::prev(it)->second->isLoaded()) {
        auto &[n, seg] = *it;
        seg = std::make_unique<Segment>(n, route_->at(n), flags_);
        QObject::connect(seg.get(), &Segment::loadFinished, this, &Replay::segmentLoadFinished);
        qDebug() << "loading segment" << n << "...";
      }
      break;
    }
  }
  const auto &cur_segment = cur->second;
  enableHttpLogging(!cur_segment->isLoaded());

  // merge the previous adjacent segment if it's loaded
  auto begin = segments_.find(cur_segment->seg_num - 1);
  if (begin == segments_.end() || !(begin->second && begin->second->isLoaded())) {
    begin = cur;
  }
  mergeSegments(begin, end);

  // free segments out of current semgnt window.
  std::for_each(segments_.begin(), begin, [](auto &e) { e.second.reset(nullptr); });
  std::for_each(end, segments_.end(), [](auto &e) { e.second.reset(nullptr); });

  // start stream thread
  if (stream_thread_ == nullptr && cur_segment->isLoaded()) {
    startStream(cur_segment.get());
  }
}

void Replay::mergeSegments(const SegmentMap::iterator &begin, const SegmentMap::iterator &end) {
  // merge 3 segments in sequence.
  std::vector<int> segments_need_merge;
  size_t new_events_size = 0;
  for (auto it = begin; it != end && it->second->isLoaded() && segments_need_merge.size() < 3; ++it) {
    segments_need_merge.push_back(it->first);
    new_events_size += it->second->log->events.size();
  }

  if (segments_need_merge != segments_merged_) {
    qDebug() << "merge segments" << segments_need_merge;
    new_events_->clear();
    new_events_->reserve(new_events_size);
    for (int n : segments_need_merge) {
      const auto &e = segments_[n]->log->events;
      auto middle = new_events_->insert(new_events_->end(), e.begin(), e.end());
      std::inplace_merge(new_events_->begin(), middle, new_events_->end(), Event::lessThan());
    }

    updateEvents([&]() {
      events_.swap(new_events_);
      segments_merged_ = segments_need_merge;
      return true;
    });
  }
}

void Replay::startStream(const Segment *cur_segment) {
  const auto &events = cur_segment->log->events;

  // get route start time from initData
  auto it = std::find_if(events.begin(), events.end(), [](auto e) { return e->which == cereal::Event::Which::INIT_DATA; });
  route_start_ts_ = it != events.end() ? (*it)->mono_time : events[0]->mono_time;
  cur_mono_time_ += route_start_ts_;

  // write CarParams
  it = std::find_if(events.begin(), events.end(), [](auto e) { return e->which == cereal::Event::Which::CAR_PARAMS; });
  if (it != events.end()) {
    auto bytes = (*it)->bytes();
    Params().put("CarParams", (const char *)bytes.begin(), bytes.size());
  } else {
    qWarning() << "failed to read CarParams from current segment";
  }

  // start camera server
  std::pair<int, int> camera_size[MAX_CAMERAS] = {};
  for (auto type : ALL_CAMERAS) {
    if (auto &fr = cur_segment->frames[type]) {
      camera_size[type] = {fr->width, fr->height};
    }
  }
  camera_server_ = std::make_unique<CameraServer>(camera_size, flags_ & REPLAY_FLAG_SEND_YUV);

  // start stream thread
  stream_thread_ = new QThread();
  QObject::connect(stream_thread_, &QThread::started, [=]() { stream(); });
  QObject::connect(stream_thread_, &QThread::finished, stream_thread_, &QThread::deleteLater);
  stream_thread_->start();
}

void Replay::publishMessage(const Event *e) {
  if (sm == nullptr) {
    auto bytes = e->bytes();
    int ret = pm->send(sockets_[e->which], (capnp::byte *)bytes.begin(), bytes.size());
    if (ret == -1) {
      qDebug() << "stop publishing" << sockets_[e->which] << "due to multiple publishers error";
      sockets_[e->which] = nullptr;
    }
  } else {
    sm->update_msgs(nanos_since_boot(), {{sockets_[e->which], e->event}});
  }
}

void Replay::publishFrame(const Event *e) {
  static const std::map<cereal::Event::Which, CameraType> cam_types{
      {cereal::Event::ROAD_ENCODE_IDX, RoadCam},
      {cereal::Event::DRIVER_ENCODE_IDX, DriverCam},
      {cereal::Event::WIDE_ROAD_ENCODE_IDX, WideRoadCam},
  };
  if ((e->which == cereal::Event::DRIVER_ENCODE_IDX && !(flags_ & REPLAY_FLAG_DCAM)) ||
      (e->which == cereal::Event::WIDE_ROAD_ENCODE_IDX && !(flags_ & REPLAY_FLAG_ECAM))) {
    return;
  }
  auto eidx = capnp::AnyStruct::Reader(e->event).getPointerSection()[0].getAs<cereal::EncodeIndex>();
  if (eidx.getType() == cereal::EncodeIndex::Type::FULL_H_E_V_C && isSegmentMerged(eidx.getSegmentNum())) {
    CameraType cam = cam_types.at(e->which);
    camera_server_->pushFrame(cam, segments_[eidx.getSegmentNum()]->frames[cam].get(), eidx);
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
      const int current_ts = currentSeconds();
      if (last_print > current_ts || (current_ts - last_print) > 5.0) {
        last_print = current_ts;
        qInfo() << "at " << current_ts << "s";
      }
      setCurrentSegment(current_ts / 60);

      // migration for pandaState -> pandaStates to keep UI working for old segments
      if (cur_which == cereal::Event::Which::PANDA_STATE_D_E_P_R_E_C_A_T_E_D) {
        MessageBuilder msg;
        auto ps = msg.initEvent().initPandaStates(1);
        ps[0].setIgnitionLine(true);
        ps[0].setPandaType(cereal::PandaState::PandaType::DOS);
        pm->send(sockets_[cereal::Event::Which::PANDA_STATES], msg);
      }

      if (cur_which < sockets_.size() && sockets_[cur_which] != nullptr) {
        // keep time
        long etime = cur_mono_time_ - evt_start_ts;
        long rtime = nanos_since_boot() - loop_start_ts;
        long behind_ns = etime - rtime;
        // if behind_ns is greater than 1 second, it means that an invalid segemnt is skipped by seeking/replaying
        if (behind_ns >= 1 * 1e9) {
          // reset start times
          evt_start_ts = cur_mono_time_;
          loop_start_ts = nanos_since_boot();
        } else if (behind_ns > 0) {
          precise_nano_sleep(behind_ns);
        }

        if (evt->frame) {
          publishFrame(evt);
        } else {
          publishMessage(evt);
        }
      }
    }
    // wait for frame to be sent before unlock.(frameReader may be deleted after unlock)
    camera_server_->waitFinish();

    if (eit == events_->end() && !(flags_ & REPLAY_FLAG_NO_LOOP)) {
      int last_segment = segments_.rbegin()->first;
      if (current_segment_ >= last_segment && isSegmentMerged(last_segment)) {
        qInfo() << "reaches the end of route, restart from beginning";
        emit seekTo(0, false);
      }
    }
  }
}
