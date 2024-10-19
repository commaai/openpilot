#include "tools/replay/replay.h"

#include <QDebug>
#include <QtConcurrent>
#include <capnp/dynamic.h>
#include <csignal>
#include "cereal/services.h"
#include "common/params.h"
#include "common/timing.h"
#include "tools/replay/util.h"

static void interrupt_sleep_handler(int signal) {}

Replay::Replay(const std::string &route, std::vector<std::string> allow, std::vector<std::string> block, SubMaster *sm_,
               uint32_t flags, const std::string &data_dir, QObject *parent) : sm(sm_), flags_(flags), QObject(parent) {
  // Register signal handler for SIGUSR1
  std::signal(SIGUSR1, interrupt_sleep_handler);

  if (!(flags_ & REPLAY_FLAG_ALL_SERVICES)) {
    block.insert(block.end(), {"uiDebug", "userFlag"});
  }

  auto event_schema = capnp::Schema::from<cereal::Event>().asStruct();
  sockets_.resize(event_schema.getUnionFields().size());
  std::vector<std::string> active_services;

  for (const auto &[name, _] : services) {
    bool in_block = std::find(block.begin(), block.end(), name) != block.end();
    bool in_allow = std::find(allow.begin(), allow.end(), name) != allow.end();
    if (!in_block && (allow.empty() || in_allow)) {
      uint16_t which = event_schema.getFieldByName(name).getProto().getDiscriminantValue();
      sockets_[which] = name.c_str();
      active_services.push_back(name);
    }
  }

  if (!allow.empty()) {
    for (int i = 0; i < sockets_.size(); ++i) {
      filters_.push_back(i == cereal::Event::Which::INIT_DATA || i == cereal::Event::Which::CAR_PARAMS || sockets_[i]);
    }
  }

  rInfo("active services: %s", join(active_services, ',').c_str());
  rInfo("loading route %s", route.c_str());

  if (sm == nullptr) {
    std::vector<const char *> socket_names;
    std::copy_if(sockets_.begin(), sockets_.end(), std::back_inserter(socket_names),
                 [](const char *name) { return name != nullptr; });
    pm = std::make_unique<PubMaster>(socket_names);
  }
  route_ = std::make_unique<Route>(route, data_dir);
}

Replay::~Replay() {
  stop();
}

void Replay::stop() {
  exit_ = true;
  if (stream_thread_ != nullptr) {
    rInfo("shutdown: in progress...");
    pauseStreamThread();
    stream_cv_.notify_one();
    stream_thread_->quit();
    stream_thread_->wait();
    stream_thread_->deleteLater();
    stream_thread_ = nullptr;
    rInfo("shutdown: done");
  }
  timeline_future.waitForFinished();
  camera_server_.reset(nullptr);
  segments_.clear();
}

bool Replay::load() {
  if (!route_->load()) {
    rError("failed to load route %s from %s", route_->name().c_str(),
           route_->dir().empty() ? "server" : route_->dir().c_str());
    return false;
  }

  for (auto &[n, f] : route_->segments()) {
    bool has_log = !f.rlog.empty() || !f.qlog.empty();
    bool has_video = !f.road_cam.empty() || !f.qcamera.empty();
    if (has_log && (has_video || hasFlag(REPLAY_FLAG_NO_VIPC))) {
      segments_.insert({n, nullptr});
    }
  }
  if (segments_.empty()) {
    rInfo("no valid segments in route: %s", route_->name().c_str());
    return false;
  }
  rInfo("load route %s with %zu valid segments", route_->name().c_str(), segments_.size());
  max_seconds_ = (segments_.rbegin()->first + 1) * 60;
  return true;
}

void Replay::start(int seconds) {
  seekTo(route_->identifier().begin_segment * 60 + seconds, false);
}

void Replay::updateEvents(const std::function<bool()> &update_events_function) {
  pauseStreamThread();
  {
    std::unique_lock lk(stream_lock_);
    events_ready_ = update_events_function();
    paused_ = user_paused_;
  }
  stream_cv_.notify_one();
}

void Replay::seekTo(double seconds, bool relative) {
  updateEvents([&]() {
    double target_time = relative ? seconds + currentSeconds() : seconds;
    target_time = std::max(double(0.0), target_time);
    int target_segment = (int)target_time / 60;
    if (segments_.count(target_segment) == 0) {
      rWarning("Can't seek to %.2f s segment %d is invalid", target_time, target_segment);
      return true;
    }
    if (target_time > max_seconds_) {
      rWarning("Can't seek to %.2f s, time is invalid", target_time);
      return true;
    }

    rInfo("Seeking to %d s, segment %d", (int)target_time, target_segment);
    current_segment_ = target_segment;
    cur_mono_time_ = route_start_ts_ + target_time * 1e9;
    seeking_to_ = target_time;
    return false;
  });

  checkSeekProgress();
  updateSegmentsCache();
}

void Replay::checkSeekProgress() {
  if (seeking_to_) {
    auto it = segments_.find(int(*seeking_to_ / 60));
    if (it != segments_.end() && it->second && it->second->isLoaded()) {
      emit seekedTo(*seeking_to_);
      seeking_to_ = std::nullopt;
      // wake up stream thread
      updateEvents([]() { return true; });
    } else {
      // Emit signal indicating the ongoing seek operation
      emit seeking(*seeking_to_);
    }
  }
}

void Replay::seekToFlag(FindFlag flag) {
  if (auto next = find(flag)) {
    seekTo(*next - 2, false);  // seek to 2 seconds before next
  }
}

void Replay::buildTimeline() {
  uint64_t engaged_begin = 0;
  bool engaged = false;

  auto alert_status = cereal::SelfdriveState::AlertStatus::NORMAL;
  auto alert_size = cereal::SelfdriveState::AlertSize::NONE;
  uint64_t alert_begin = 0;
  std::string alert_type;

  const TimelineType timeline_types[] = {
    [(int)cereal::SelfdriveState::AlertStatus::NORMAL] = TimelineType::AlertInfo,
    [(int)cereal::SelfdriveState::AlertStatus::USER_PROMPT] = TimelineType::AlertWarning,
    [(int)cereal::SelfdriveState::AlertStatus::CRITICAL] = TimelineType::AlertCritical,
  };

  const auto &route_segments = route_->segments();
  for (auto it = route_segments.cbegin(); it != route_segments.cend() && !exit_; ++it) {
    std::shared_ptr<LogReader> log(new LogReader());
    if (!log->load(it->second.qlog, &exit_, !hasFlag(REPLAY_FLAG_NO_FILE_CACHE), 0, 3) || log->events.empty()) continue;

    std::vector<std::tuple<double, double, TimelineType>> timeline;
    for (const Event &e : log->events) {
      if (e.which == cereal::Event::Which::SELFDRIVE_STATE) {
        capnp::FlatArrayMessageReader reader(e.data);
        auto event = reader.getRoot<cereal::Event>();
        auto cs = event.getSelfdriveState();

        if (engaged != cs.getEnabled()) {
          if (engaged) {
            timeline.push_back({toSeconds(engaged_begin), toSeconds(e.mono_time), TimelineType::Engaged});
          }
          engaged_begin = e.mono_time;
          engaged = cs.getEnabled();
        }

        if (alert_type != cs.getAlertType().cStr() || alert_status != cs.getAlertStatus()) {
          if (!alert_type.empty() && alert_size != cereal::SelfdriveState::AlertSize::NONE) {
            timeline.push_back({toSeconds(alert_begin), toSeconds(e.mono_time), timeline_types[(int)alert_status]});
          }
          alert_begin = e.mono_time;
          alert_type = cs.getAlertType().cStr();
          alert_size = cs.getAlertSize();
          alert_status = cs.getAlertStatus();
        }
      } else if (e.which == cereal::Event::Which::USER_FLAG) {
        timeline.push_back({toSeconds(e.mono_time), toSeconds(e.mono_time), TimelineType::UserFlag});
      }
    }

    if (it->first == route_segments.rbegin()->first) {
      if (engaged) {
        timeline.push_back({toSeconds(engaged_begin), toSeconds(log->events.back().mono_time), TimelineType::Engaged});
      }
      if (!alert_type.empty() && alert_size != cereal::SelfdriveState::AlertSize::NONE) {
        timeline.push_back({toSeconds(alert_begin), toSeconds(log->events.back().mono_time), timeline_types[(int)alert_status]});
      }

      max_seconds_ = std::ceil(toSeconds(log->events.back().mono_time));
      emit minMaxTimeChanged(route_segments.cbegin()->first * 60.0, max_seconds_);
    }
    {
      std::lock_guard lk(timeline_lock);
      timeline_.insert(timeline_.end(), timeline.begin(), timeline.end());
      std::sort(timeline_.begin(), timeline_.end(), [](auto &l, auto &r) { return std::get<2>(l) < std::get<2>(r); });
    }
    emit qLogLoaded(log);
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
  if (user_paused_ != pause) {
    pauseStreamThread();
    {
      std::unique_lock lk(stream_lock_);
      rWarning("%s at %.2f s", pause ? "paused..." : "resuming", currentSeconds());
      paused_ = user_paused_ = pause;
    }
    stream_cv_.notify_one();
  }
}

void Replay::pauseStreamThread() {
  paused_ = true;
  // Send SIGUSR1 to interrupt clock_nanosleep
  if (stream_thread_ && stream_thread_id) {
    pthread_kill(stream_thread_id, SIGUSR1);
  }
}

void Replay::segmentLoadFinished(bool success) {
  if (!success) {
    Segment *seg = qobject_cast<Segment *>(sender());
    rWarning("failed to load segment %d, removing it from current replay list", seg->seg_num);
    updateEvents([&]() {
      segments_.erase(seg->seg_num);
      return !segments_.empty();
    });
  }
  updateSegmentsCache();
}

void Replay::updateSegmentsCache() {
  auto cur = segments_.lower_bound(current_segment_.load());
  if (cur == segments_.end()) return;

  // Calculate the range of segments to load
  auto begin = std::prev(cur, std::min<int>(segment_cache_limit / 2, std::distance(segments_.begin(), cur)));
  auto end = std::next(begin, std::min<int>(segment_cache_limit, std::distance(begin, segments_.end())));
  begin = std::prev(end, std::min<int>(segment_cache_limit, std::distance(segments_.begin(), end)));

  loadSegmentInRange(begin, cur, end);
  mergeSegments(begin, end);

  // free segments out of current semgnt window.
  std::for_each(segments_.begin(), begin, [](auto &e) { e.second.reset(nullptr); });
  std::for_each(end, segments_.end(), [](auto &e) { e.second.reset(nullptr); });

  // start stream thread
  const auto &cur_segment = cur->second;
  if (stream_thread_ == nullptr && cur_segment->isLoaded()) {
    startStream(cur_segment.get());
  }
}

void Replay::loadSegmentInRange(SegmentMap::iterator begin, SegmentMap::iterator cur, SegmentMap::iterator end) {
  auto loadNextSegment = [this](auto first, auto last) {
    auto it = std::find_if(first, last, [](const auto &seg_it) { return !seg_it.second || !seg_it.second->isLoaded(); });
    if (it != last && !it->second) {
      rDebug("loading segment %d...", it->first);
      it->second = std::make_unique<Segment>(it->first, route_->at(it->first), flags_, filters_);
      QObject::connect(it->second.get(), &Segment::loadFinished, this, &Replay::segmentLoadFinished);
      return true;
    }
    return false;
  };

  // Try loading forward segments, then reverse segments
  if (!loadNextSegment(cur, end)) {
    loadNextSegment(std::make_reverse_iterator(cur), std::make_reverse_iterator(begin));
  }
}

void Replay::mergeSegments(const SegmentMap::iterator &begin, const SegmentMap::iterator &end) {
  std::set<int> segments_to_merge;
  size_t new_events_size = 0;
  for (auto it = begin; it != end; ++it) {
    if (it->second && it->second->isLoaded()) {
      segments_to_merge.insert(it->first);
      new_events_size += it->second->log->events.size();
    }
  }

  if (segments_to_merge == merged_segments_) return;

  rDebug("merge segments %s", std::accumulate(segments_to_merge.begin(), segments_to_merge.end(), std::string{},
    [](auto & a, int b) { return a + (a.empty() ? "" : ", ") + std::to_string(b); }).c_str());

  std::vector<Event> new_events;
  new_events.reserve(new_events_size);

  // Merge events from segments_to_merge into new_events
  for (int n : segments_to_merge) {
    size_t size = new_events.size();
    const auto &events = segments_.at(n)->log->events;
    std::copy_if(events.begin(), events.end(), std::back_inserter(new_events),
                  [this](const Event &e) { return e.which < sockets_.size() && sockets_[e.which] != nullptr; });
    std::inplace_merge(new_events.begin(), new_events.begin() + size, new_events.end());
  }

  if (stream_thread_) {
    emit segmentsMerged();
  }

  updateEvents([&]() {
    events_.swap(new_events);
    merged_segments_ = segments_to_merge;
    // Wake up the stream thread if the current segment is loaded or invalid.
    return !seeking_to_ && (isSegmentMerged(current_segment_) || (segments_.count(current_segment_) == 0));
  });
  checkSeekProgress();
}

void Replay::startStream(const Segment *cur_segment) {
  const auto &events = cur_segment->log->events;
  route_start_ts_ = events.front().mono_time;
  cur_mono_time_ += route_start_ts_ - 1;

  // get datetime from INIT_DATA, fallback to datetime in the route name
  route_date_time_ = route()->datetime();
  auto it = std::find_if(events.cbegin(), events.cend(),
                         [](const Event &e) { return e.which == cereal::Event::Which::INIT_DATA; });
  if (it != events.cend()) {
    capnp::FlatArrayMessageReader reader(it->data);
    auto event = reader.getRoot<cereal::Event>();
    uint64_t wall_time = event.getInitData().getWallTimeNanos();
    if (wall_time > 0) {
      route_date_time_ = QDateTime::fromMSecsSinceEpoch(wall_time / 1e6);
    }
  }

  // write CarParams
  it = std::find_if(events.begin(), events.end(), [](const Event &e) { return e.which == cereal::Event::Which::CAR_PARAMS; });
  if (it != events.end()) {
    capnp::FlatArrayMessageReader reader(it->data);
    auto event = reader.getRoot<cereal::Event>();
    car_fingerprint_ = event.getCarParams().getCarFingerprint();
    capnp::MallocMessageBuilder builder;
    builder.setRoot(event.getCarParams());
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
  QObject::connect(stream_thread_, &QThread::started, [=]() { streamThread(); });
  stream_thread_->start();

  timeline_future = QtConcurrent::run(this, &Replay::buildTimeline);
  emit streamStarted();
}

void Replay::publishMessage(const Event *e) {
  if (event_filter && event_filter(e, filter_opaque)) return;

  if (sm == nullptr) {
    auto bytes = e->data.asBytes();
    int ret = pm->send(sockets_[e->which], (capnp::byte *)bytes.begin(), bytes.size());
    if (ret == -1) {
      rWarning("stop publishing %s due to multiple publishers error", sockets_[e->which]);
      sockets_[e->which] = nullptr;
    }
  } else {
    capnp::FlatArrayMessageReader reader(e->data);
    auto event = reader.getRoot<cereal::Event>();
    sm->update_msgs(nanos_since_boot(), {{sockets_[e->which], event}});
  }
}

void Replay::publishFrame(const Event *e) {
  CameraType cam;
  switch (e->which) {
    case cereal::Event::ROAD_ENCODE_IDX: cam = RoadCam; break;
    case cereal::Event::DRIVER_ENCODE_IDX: cam = DriverCam; break;
    case cereal::Event::WIDE_ROAD_ENCODE_IDX: cam = WideRoadCam; break;
    default: return;  // Invalid event type
  }

  if ((cam == DriverCam && !hasFlag(REPLAY_FLAG_DCAM)) || (cam == WideRoadCam && !hasFlag(REPLAY_FLAG_ECAM)))
    return;  // Camera isdisabled

  if (isSegmentMerged(e->eidx_segnum)) {
    auto &segment = segments_.at(e->eidx_segnum);
    if (auto &frame = segment->frames[cam]; frame) {
      camera_server_->pushFrame(cam, frame.get(), e);
    }
  }
}

void Replay::streamThread() {
  stream_thread_id = pthread_self();
  cereal::Event::Which cur_which = cereal::Event::Which::INIT_DATA;
  std::unique_lock lk(stream_lock_);

  while (true) {
    stream_cv_.wait(lk, [=]() { return exit_ || ( events_ready_ && !paused_); });
    if (exit_) break;

    Event event(cur_which, cur_mono_time_, {});
    auto first = std::upper_bound(events_.cbegin(), events_.cend(), event);
    if (first == events_.cend()) {
      rInfo("waiting for events...");
      events_ready_ = false;
      continue;
    }

    auto it = publishEvents(first, events_.cend());

    // Ensure frames are sent before unlocking to prevent race conditions
    if (camera_server_) {
      camera_server_->waitForSent();
    }

    if (it != events_.cend()) {
      cur_which = it->which;
    } else if (!hasFlag(REPLAY_FLAG_NO_LOOP)) {
      // Check for loop end and restart if necessary
      int last_segment = segments_.rbegin()->first;
      if (current_segment_ >= last_segment && isSegmentMerged(last_segment)) {
        rInfo("reaches the end of route, restart from beginning");
        QMetaObject::invokeMethod(this, std::bind(&Replay::seekTo, this, minSeconds(), false), Qt::QueuedConnection);
      }
    }
  }
}

std::vector<Event>::const_iterator Replay::publishEvents(std::vector<Event>::const_iterator first,
                                                         std::vector<Event>::const_iterator last) {
  uint64_t evt_start_ts = cur_mono_time_;
  uint64_t loop_start_ts = nanos_since_boot();
  double prev_replay_speed = speed_;

  for (; !paused_ && first != last; ++first) {
    const Event &evt = *first;
    int segment = toSeconds(evt.mono_time) / 60;

    if (current_segment_ != segment) {
      current_segment_ = segment;
      QMetaObject::invokeMethod(this, &Replay::updateSegmentsCache, Qt::QueuedConnection);
    }

     // Skip events if socket is not present
    if (!sockets_[evt.which]) continue;

    cur_mono_time_ = evt.mono_time;
    const uint64_t current_nanos = nanos_since_boot();
    const int64_t time_diff = (evt.mono_time - evt_start_ts) / speed_ - (current_nanos - loop_start_ts);

    // Reset timestamps for potential synchronization issues:
    // - A negative time_diff may indicate slow execution or system wake-up,
    // - A time_diff exceeding 1 second suggests a skipped segment.
    if ((time_diff < -1e9 || time_diff >= 1e9) || speed_ != prev_replay_speed) {
      evt_start_ts = evt.mono_time;
      loop_start_ts = current_nanos;
      prev_replay_speed = speed_;
    } else if (time_diff > 0) {
      precise_nano_sleep(time_diff, paused_);
    }

    if (paused_) break;

    if (evt.eidx_segnum == -1) {
      publishMessage(&evt);
    } else if (camera_server_) {
      if (speed_ > 1.0) {
        camera_server_->waitForSent();
      }
      publishFrame(&evt);
    }
  }

  return first;
}
