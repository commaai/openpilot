#include "tools/replay/replay.h"

#include <QDebug>
#include <QtConcurrent>

#include <capnp/dynamic.h>
#include "cereal/services.h"
#include "common/params.h"
#include "common/timing.h"
#include "system/hardware/hw.h"
#include "tools/replay/util.h"

Replay::Replay(QString route, QStringList allow, QStringList block, QStringList base_blacklist, SubMaster *sm_, uint32_t flags, QString data_dir, QObject *parent)
    : sm(sm_), flags_(flags), QObject(parent) {
  std::vector<const char *> s;
  auto event_struct = capnp::Schema::from<cereal::Event>().asStruct();
  sockets_.resize(event_struct.getUnionFields().size());
  for (const auto &it : services) {
    auto name = it.second.name.c_str();
    uint16_t which = event_struct.getFieldByName(name).getProto().getDiscriminantValue();
    if ((which == cereal::Event::Which::UI_DEBUG || which == cereal::Event::Which::USER_FLAG) &&
        !(flags & REPLAY_FLAG_ALL_SERVICES) &&
        !allow.contains(name)) {
      continue;
    }

    if ((allow.empty() || allow.contains(name)) && !block.contains(name)) {
      sockets_[which] = name;
      if (!allow.empty() || !block.empty()) {
        allow_list.insert((cereal::Event::Which)which);
      }
      s.push_back(name);
    }
  }

  if (!allow_list.empty()) {
    // the following events are needed for replay to work properly.
    allow_list.insert(cereal::Event::Which::INIT_DATA);
    allow_list.insert(cereal::Event::Which::CAR_PARAMS);
    if (sockets_[cereal::Event::Which::PANDA_STATES] != nullptr) {
      allow_list.insert(cereal::Event::Which::PANDA_STATE_D_E_P_R_E_C_A_T_E_D);
    }
  }

  qDebug() << "services " << s;
  qDebug() << "loading route " << route;

  if (sm == nullptr) {
    pm = std::make_unique<PubMaster>(s);
  }
  route_ = std::make_unique<Route>(route, data_dir);
  events_ = std::make_unique<std::vector<Event *>>();
  new_events_ = std::make_unique<std::vector<Event *>>();
}

Replay::~Replay() {
  stop();
}

void Replay::stop() {
  if (!stream_thread_ && segments_.empty()) return;

  rInfo("shutdown: in progress...");
  if (stream_thread_ != nullptr) {
    exit_ = updating_events_ = true;
    stream_cv_.notify_one();
    stream_thread_->quit();
    stream_thread_->wait();
    stream_thread_ = nullptr;
  }
  camera_server_.reset(nullptr);
  timeline_future.waitForFinished();
  segments_.clear();
  rInfo("shutdown: done");
}

bool Replay::load() {
  if (!route_->load()) {
    qCritical() << "failed to load route" << route_->name()
                << "from" << (route_->dir().isEmpty() ? "server" : route_->dir());
    return false;
  }

  for (auto &[n, f] : route_->segments()) {
    bool has_log = !f.rlog.isEmpty() || !f.qlog.isEmpty();
    bool has_video = !f.road_cam.isEmpty() || !f.qcamera.isEmpty();
    if (has_log && (has_video || hasFlag(REPLAY_FLAG_NO_VIPC))) {
      segments_.insert({n, nullptr});
    }
  }
  if (segments_.empty()) {
    qCritical() << "no valid segments in route" << route_->name();
    return false;
  }
  rInfo("load route %s with %zu valid segments", qPrintable(route_->name()), segments_.size());
  return true;
}

void Replay::start(int seconds) {
  seekTo(route_->identifier().segment_id * 60 + seconds, false);
}

void Replay::updateEvents(const std::function<bool()> &lambda) {
  // set updating_events to true to force stream thread release the lock and wait for events_updated.
  updating_events_ = true;
  {
    std::unique_lock lk(stream_lock_);
    events_updated_ = lambda();
    updating_events_ = false;
  }
  stream_cv_.notify_one();
}

void Replay::seekTo(double seconds, bool relative) {
  seconds = relative ? seconds + currentSeconds() : seconds;
  updateEvents([&]() {
    seconds = std::max(double(0.0), seconds);
    int seg = (int)seconds / 60;
    if (segments_.find(seg) == segments_.end()) {
      rWarning("can't seek to %d s segment %d is invalid", seconds, seg);
      return true;
    }

    rInfo("seeking to %d s, segment %d", (int)seconds, seg);
    current_segment_ = seg;
    cur_mono_time_ = route_start_ts_ + seconds * 1e9;
    emit seekedTo(seconds);
    return isSegmentMerged(seg);
  });
  queueSegment();
}

void Replay::seekToFlag(FindFlag flag) {
  if (auto next = find(flag)) {
    seekTo(*next - 2, false);  // seek to 2 seconds before next
  }
}

void Replay::buildTimeline() {
  uint64_t engaged_begin = 0;
  bool engaged = false;

  auto alert_status = cereal::ControlsState::AlertStatus::NORMAL;
  auto alert_size = cereal::ControlsState::AlertSize::NONE;
  uint64_t alert_begin = 0;
  std::string alert_type;

  const TimelineType timeline_types[] = {
    [(int)cereal::ControlsState::AlertStatus::NORMAL] = TimelineType::AlertInfo,
    [(int)cereal::ControlsState::AlertStatus::USER_PROMPT] = TimelineType::AlertWarning,
    [(int)cereal::ControlsState::AlertStatus::CRITICAL] = TimelineType::AlertCritical,
  };

  const auto &route_segments = route_->segments();
  for (auto it = route_segments.cbegin(); it != route_segments.cend() && !exit_; ++it) {
    LogReader log;
    if (!log.load(it->second.qlog.toStdString(), &exit_,
                  {cereal::Event::Which::CONTROLS_STATE, cereal::Event::Which::USER_FLAG},
                  !hasFlag(REPLAY_FLAG_NO_FILE_CACHE), 0, 3)) continue;

    for (const Event *e : log.events) {
      if (e->which == cereal::Event::Which::CONTROLS_STATE) {
        auto cs = e->event.getControlsState();

        if (engaged != cs.getEnabled()) {
          if (engaged) {
            std::lock_guard lk(timeline_lock);
            timeline.push_back({toSeconds(engaged_begin), toSeconds(e->mono_time), TimelineType::Engaged});
          }
          engaged_begin = e->mono_time;
          engaged = cs.getEnabled();
        }

        if (alert_type != cs.getAlertType().cStr() || alert_status != cs.getAlertStatus()) {
          if (!alert_type.empty() && alert_size != cereal::ControlsState::AlertSize::NONE) {
            std::lock_guard lk(timeline_lock);
            timeline.push_back({toSeconds(alert_begin), toSeconds(e->mono_time), timeline_types[(int)alert_status]});
          }
          alert_begin = e->mono_time;
          alert_type = cs.getAlertType().cStr();
          alert_size = cs.getAlertSize();
          alert_status = cs.getAlertStatus();
        }
      } else if (e->which == cereal::Event::Which::USER_FLAG) {
        std::lock_guard lk(timeline_lock);
        timeline.push_back({toSeconds(e->mono_time), toSeconds(e->mono_time), TimelineType::UserFlag});
      }
    }
  }
}

std::optional<uint64_t> Replay::find(FindFlag flag) {
  int cur_ts = currentSeconds();
  for (auto [start_ts, end_ts, type] : getTimeline()) {
    if (type == TimelineType::Engaged) {
      if (flag == FindFlag::nextEngagement && start_ts > cur_ts) {
        return start_ts;
      } else if (flag == FindFlag::nextDisEngagement && end_ts > cur_ts) {
        return end_ts;
      }
    } else if (start_ts > cur_ts) {
      if ((flag == FindFlag::nextUserFlag && type == TimelineType::UserFlag) ||
          (flag == FindFlag::nextInfo && type == TimelineType::AlertInfo) ||
          (flag == FindFlag::nextWarning && type == TimelineType::AlertWarning) ||
          (flag == FindFlag::nextCritical && type == TimelineType::AlertCritical)) {
        return start_ts;
      }
    }
  }
  return std::nullopt;
}

void Replay::pause(bool pause) {
  updateEvents([=]() {
    rWarning("%s at %.2f s", pause ? "paused..." : "resuming", currentSeconds());
    paused_ = pause;
    return true;
  });
}

void Replay::setCurrentSegment(int n) {
  if (current_segment_.exchange(n) != n) {
    QMetaObject::invokeMethod(this, &Replay::queueSegment, Qt::QueuedConnection);
  }
}

void Replay::segmentLoadFinished(bool success) {
  if (!success) {
    Segment *seg = qobject_cast<Segment *>(sender());
    rWarning("failed to load segment %d, removing it from current replay list", seg->seg_num);
    updateEvents([&]() {
      segments_.erase(seg->seg_num);
      return true;
    });
  }
  queueSegment();
}

void Replay::queueSegment() {
  if (segments_.empty()) return;

  SegmentMap::iterator begin, cur;
  begin = cur = segments_.lower_bound(std::min(current_segment_.load(), segments_.rbegin()->first));
  int distance = std::max<int>(std::ceil(segment_cache_limit / 2.0) - 1, segment_cache_limit - std::distance(cur, segments_.end()));
  for (int i = 0; begin != segments_.begin() && i < distance; ++i) {
    --begin;
  }
  auto end = begin;
  for (int i = 0; end != segments_.end() && i < segment_cache_limit; ++i) {
    ++end;
  }

  // load one segment at a time
  for (auto it = cur; it != end; ++it) {
    auto &[n, seg] = *it;
    if ((seg && !seg->isLoaded()) || !seg) {
      if (!seg) {
        rDebug("loading segment %d...", n);
        seg = std::make_unique<Segment>(n, route_->at(n), flags_, allow_list);
        QObject::connect(seg.get(), &Segment::loadFinished, this, &Replay::segmentLoadFinished);
      }
      break;
    }
  }

  mergeSegments(begin, end);

  // free segments out of current semgnt window.
  std::for_each(segments_.begin(), begin, [](auto &e) { e.second.reset(nullptr); });
  std::for_each(end, segments_.end(), [](auto &e) { e.second.reset(nullptr); });

  // start stream thread
  const auto &cur_segment = cur->second;
  if (stream_thread_ == nullptr && cur_segment->isLoaded()) {
    startStream(cur_segment.get());
    emit streamStarted();
  }
}

void Replay::mergeSegments(const SegmentMap::iterator &begin, const SegmentMap::iterator &end) {
  std::vector<int> segments_need_merge;
  size_t new_events_size = 0;
  for (auto it = begin; it != end; ++it) {
    if (it->second && it->second->isLoaded()) {
      segments_need_merge.push_back(it->first);
      new_events_size += it->second->log->events.size();
    }
  }

  if (segments_need_merge != segments_merged_) {
    std::string s;
    for (int i = 0; i < segments_need_merge.size(); ++i) {
      s += std::to_string(segments_need_merge[i]);
      if (i != segments_need_merge.size() - 1) s += ", ";
    }
    rDebug("merge segments %s", s.c_str());
    new_events_->clear();
    new_events_->reserve(new_events_size);
    for (int n : segments_need_merge) {
      const auto &e = segments_[n]->log->events;
      if (e.size() > 0) {
        auto insert_from = e.begin();
        if (new_events_->size() > 0 && (*insert_from)->which == cereal::Event::Which::INIT_DATA) ++insert_from;
        auto middle = new_events_->insert(new_events_->end(), insert_from, e.end());
        std::inplace_merge(new_events_->begin(), middle, new_events_->end(), Event::lessThan());
      }
    }

    if (stream_thread_) {
      emit segmentsMerged();
    }
    updateEvents([&]() {
      events_.swap(new_events_);
      segments_merged_ = segments_need_merge;
      // Do not wake up the stream thread if the current segment has not been merged.
      return isSegmentMerged(current_segment_) || (segments_.count(current_segment_) == 0);
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
    car_fingerprint_ = (*it)->event.getCarParams().getCarFingerprint();
    capnp::MallocMessageBuilder builder;
    builder.setRoot((*it)->event.getCarParams());
    auto words = capnp::messageToFlatArray(builder);
    auto bytes = words.asBytes();
    Params().put("CarParams", (const char *)bytes.begin(), bytes.size());
    Params().put("CarParamsPersistent", (const char *)bytes.begin(), bytes.size());
  } else {
    rWarning("failed to read CarParams from current segment");
  }

  // start camera server
  if (!hasFlag(REPLAY_FLAG_NO_VIPC)) {
    std::pair<int, int> camera_size[MAX_CAMERAS] = {};
    for (auto type : ALL_CAMERAS) {
      if (auto &fr = cur_segment->frames[type]) {
        camera_size[type] = {fr->width, fr->height};
      }
    }
    camera_server_ = std::make_unique<CameraServer>(camera_size);
  }

  emit segmentsMerged();
  // start stream thread
  stream_thread_ = new QThread();
  QObject::connect(stream_thread_, &QThread::started, [=]() { stream(); });
  QObject::connect(stream_thread_, &QThread::finished, stream_thread_, &QThread::deleteLater);
  stream_thread_->start();

  timeline_future = QtConcurrent::run(this, &Replay::buildTimeline);
}

void Replay::publishMessage(const Event *e) {
  if (event_filter && event_filter(e, filter_opaque)) return;

  if (sm == nullptr) {
    auto bytes = e->bytes();
    int ret = pm->send(sockets_[e->which], (capnp::byte *)bytes.begin(), bytes.size());
    if (ret == -1) {
      rWarning("stop publishing %s due to multiple publishers error", sockets_[e->which]);
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
  if ((e->which == cereal::Event::DRIVER_ENCODE_IDX && !hasFlag(REPLAY_FLAG_DCAM)) ||
      (e->which == cereal::Event::WIDE_ROAD_ENCODE_IDX && !hasFlag(REPLAY_FLAG_ECAM))) {
    return;
  }
  auto eidx = capnp::AnyStruct::Reader(e->event).getPointerSection()[0].getAs<cereal::EncodeIndex>();
  if (eidx.getType() == cereal::EncodeIndex::Type::FULL_H_E_V_C && isSegmentMerged(eidx.getSegmentNum())) {
    CameraType cam = cam_types.at(e->which);
    camera_server_->pushFrame(cam, segments_[eidx.getSegmentNum()]->frames[cam].get(), eidx);
  }
}

void Replay::stream() {
  cereal::Event::Which cur_which = cereal::Event::Which::INIT_DATA;
  double prev_replay_speed = 1.0;
  std::unique_lock lk(stream_lock_);

  while (true) {
    stream_cv_.wait(lk, [=]() { return exit_ || (events_updated_ && !paused_); });
    events_updated_ = false;
    if (exit_) break;

    Event cur_event(cur_which, cur_mono_time_);
    auto eit = std::upper_bound(events_->begin(), events_->end(), &cur_event, Event::lessThan());
    if (eit == events_->end()) {
      rInfo("waiting for events...");
      continue;
    }

    uint64_t evt_start_ts = cur_mono_time_;
    uint64_t loop_start_ts = nanos_since_boot();

    for (auto end = events_->end(); !updating_events_ && eit != end; ++eit) {
      const Event *evt = (*eit);
      cur_which = evt->which;
      cur_mono_time_ = evt->mono_time;
      setCurrentSegment(toSeconds(cur_mono_time_) / 60);

      // migration for pandaState -> pandaStates to keep UI working for old segments
      if (cur_which == cereal::Event::Which::PANDA_STATE_D_E_P_R_E_C_A_T_E_D &&
          sockets_[cereal::Event::Which::PANDA_STATES] != nullptr) {
        MessageBuilder msg;
        auto ps = msg.initEvent().initPandaStates(1);
        ps[0].setIgnitionLine(true);
        ps[0].setPandaType(cereal::PandaState::PandaType::DOS);
        pm->send(sockets_[cereal::Event::Which::PANDA_STATES], msg);
      }

      if (cur_which < sockets_.size() && sockets_[cur_which] != nullptr) {
        // keep time
        long etime = (cur_mono_time_ - evt_start_ts) / speed_;
        long rtime = nanos_since_boot() - loop_start_ts;
        long behind_ns = etime - rtime;
        // if behind_ns is greater than 1 second, it means that an invalid segemnt is skipped by seeking/replaying
        if (behind_ns >= 1 * 1e9 || speed_ != prev_replay_speed) {
          // reset event start times
          evt_start_ts = cur_mono_time_;
          loop_start_ts = nanos_since_boot();
          prev_replay_speed = speed_;
        } else if (behind_ns > 0 && !hasFlag(REPLAY_FLAG_FULL_SPEED)) {
          precise_nano_sleep(behind_ns);
        }

        if (!evt->frame) {
          publishMessage(evt);
        } else if (camera_server_) {
          if (hasFlag(REPLAY_FLAG_FULL_SPEED)) {
            camera_server_->waitForSent();
          }
          publishFrame(evt);
        }
      }
    }
    // wait for frame to be sent before unlock.(frameReader may be deleted after unlock)
    if (camera_server_) {
      camera_server_->waitForSent();
    }

    if (eit == events_->end() && !hasFlag(REPLAY_FLAG_NO_LOOP)) {
      int last_segment = segments_.empty() ? 0 : segments_.rbegin()->first;
      if (current_segment_ >= last_segment && isSegmentMerged(last_segment)) {
        rInfo("reaches the end of route, restart from beginning");
        QMetaObject::invokeMethod(this, std::bind(&Replay::seekTo, this, 0, false), Qt::QueuedConnection);
      }
    }
  }
}
