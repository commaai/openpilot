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

Replay::Replay(QString route, QStringList allow, QStringList block, SubMaster *sm_,
               uint32_t flags, QString data_dir, QObject *parent) : sm(sm_), flags_(flags), QObject(parent) {
  // Register signal handler for SIGUSR1
  std::signal(SIGUSR1, interrupt_sleep_handler);

  if (!(flags_ & REPLAY_FLAG_ALL_SERVICES)) {
    block << "uiDebug" << "userFlag";
  }
  auto event_struct = capnp::Schema::from<cereal::Event>().asStruct();
  sockets_.resize(event_struct.getUnionFields().size());
  for (const auto &[name, _] : services) {
    if (!block.contains(name.c_str()) && (allow.empty() || allow.contains(name.c_str()))) {
      uint16_t which = event_struct.getFieldByName(name).getProto().getDiscriminantValue();
      sockets_[which] = name.c_str();
    }
  }
  if (!allow.isEmpty()) {
    for (int i = 0; i < sockets_.size(); ++i) {
      filters_.push_back(i == cereal::Event::Which::INIT_DATA || i == cereal::Event::Which::CAR_PARAMS || sockets_[i]);
    }
  }

  std::vector<const char *> s;
  std::copy_if(sockets_.begin(), sockets_.end(), std::back_inserter(s),
               [](const char *name) { return name != nullptr; });
  qDebug() << "services " << s;
  qDebug() << "loading route " << route;

  if (sm == nullptr) {
    pm = std::make_unique<PubMaster>(s);
  }
  route_ = std::make_unique<Route>(route, data_dir);
}

Replay::~Replay() {
  stop();
}

void Replay::stop() {
  if (!stream_thread_ && segments_.empty()) return;

  rInfo("shutdown: in progress...");
  if (stream_thread_ != nullptr) {
    exit_ = true;
    paused_ = true;
    stream_cv_.notify_one();
    stream_thread_->quit();
    stream_thread_->wait();
    delete stream_thread_;
    stream_thread_ = nullptr;
  }
  timeline_future.waitForFinished();
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
    seeking_to_seconds_ = relative ? seconds + currentSeconds() : seconds;
    seeking_to_seconds_ = std::max(double(0.0), seeking_to_seconds_);
    int target_segment = (int)seeking_to_seconds_ / 60;
    if (segments_.count(target_segment) == 0) {
      rWarning("can't seek to %d s segment %d is invalid", (int)seeking_to_seconds_, target_segment);
      return true;
    }

    rInfo("seeking to %d s, segment %d", (int)seeking_to_seconds_, target_segment);
    current_segment_ = target_segment;
    cur_mono_time_ = route_start_ts_ + seeking_to_seconds_ * 1e9;
    bool segment_merged = isSegmentMerged(target_segment);
    if (segment_merged) {
      emit seekedTo(seeking_to_seconds_);
      // Reset seeking_to_seconds_ to indicate completion of seek
      seeking_to_seconds_ = -1;
    }
    return segment_merged;
  });
  updateSegmentsCache();
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
    std::shared_ptr<LogReader> log(new LogReader());
    if (!log->load(it->second.qlog.toStdString(), &exit_, !hasFlag(REPLAY_FLAG_NO_FILE_CACHE), 0, 3)) continue;

    for (const Event &e : log->events) {
      if (e.which == cereal::Event::Which::CONTROLS_STATE) {
        capnp::FlatArrayMessageReader reader(e.data);
        auto event = reader.getRoot<cereal::Event>();
        auto cs = event.getControlsState();

        if (engaged != cs.getEnabled()) {
          if (engaged) {
            std::lock_guard lk(timeline_lock);
            timeline.push_back({toSeconds(engaged_begin), toSeconds(e.mono_time), TimelineType::Engaged});
          }
          engaged_begin = e.mono_time;
          engaged = cs.getEnabled();
        }

        if (alert_type != cs.getAlertType().cStr() || alert_status != cs.getAlertStatus()) {
          if (!alert_type.empty() && alert_size != cereal::ControlsState::AlertSize::NONE) {
            std::lock_guard lk(timeline_lock);
            timeline.push_back({toSeconds(alert_begin), toSeconds(e.mono_time), timeline_types[(int)alert_status]});
          }
          alert_begin = e.mono_time;
          alert_type = cs.getAlertType().cStr();
          alert_size = cs.getAlertSize();
          alert_status = cs.getAlertStatus();
        }
      } else if (e.which == cereal::Event::Which::USER_FLAG) {
        std::lock_guard lk(timeline_lock);
        timeline.push_back({toSeconds(e.mono_time), toSeconds(e.mono_time), TimelineType::UserFlag});
      }
    }
    std::sort(timeline.begin(), timeline.end(), [](auto &l, auto &r) { return std::get<2>(l) < std::get<2>(r); });
    emit qLogLoaded(it->first, log);
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
    emit streamStarted();
  }
}

void Replay::loadSegmentInRange(SegmentMap::iterator begin, SegmentMap::iterator cur, SegmentMap::iterator end) {
  auto loadNext = [this](auto begin, auto end) {
    auto it = std::find_if(begin, end, [](const auto &seg_it) { return !seg_it.second || !seg_it.second->isLoaded(); });
    if (it != end && !it->second) {
      rDebug("loading segment %d...", it->first);
      it->second = std::make_unique<Segment>(it->first, route_->at(it->first), flags_, filters_);
      QObject::connect(it->second.get(), &Segment::loadFinished, this, &Replay::segmentLoadFinished);
      return true;
    }
    return false;
  };

  // Load forward segments, then try reverse
  if (!loadNext(cur, end)) {
    loadNext(std::make_reverse_iterator(cur), segments_.rend());
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
    // Check if seeking is in progress
    int target_segment = int(seeking_to_seconds_ / 60);
    if (seeking_to_seconds_ >= 0 && segments_to_merge.count(target_segment) > 0) {
      emit seekedTo(seeking_to_seconds_);
      seeking_to_seconds_ = -1;  // Reset seeking_to_seconds_ to indicate completion of seek
    }
    // Wake up the stream thread if the current segment is loaded or invalid.
    return isSegmentMerged(current_segment_) || (segments_.count(current_segment_) == 0);
  });
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
        QMetaObject::invokeMethod(this, std::bind(&Replay::seekTo, this, 0, false), Qt::QueuedConnection);
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
      precise_nano_sleep(time_diff);
    }

    if (paused_) break;

    cur_mono_time_ = evt.mono_time;
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
