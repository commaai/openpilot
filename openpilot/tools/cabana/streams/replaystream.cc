#include "tools/cabana/streams/replaystream.h"

#include <string>
#include <stdexcept>
#include <limits>

#include "common/timing.h"
#include "common/util.h"
#include "tools/cabana/settings.h"

ReplayStream::ReplayStream() {
  setenv("COMMA_CACHE", "/tmp/comma_download_cache", 1);

  op_prefix = std::make_unique<OpenpilotPrefix>();

  settings_connection_ = settings.changed.connect([this]() {
    if (replay) replay->setSegmentCacheLimit(settings.max_cached_minutes);
  });
}

ReplayStream::~ReplayStream() {
  cancelWaits();
  if (replay) replay->stop();
}

// runs on replay's merge thread: a segment of CAN data takes ~30 ms to parse and group, which dropped
// frames when it ran on the main thread. Only the sorted insert and the merged signal need the main thread.
void ReplayStream::mergeSegments() {
  auto event_data = replay->getEventData();
  for (const auto &[n, seg] : event_data->segments) {
    if (!processed_segments.count(n)) {
      processed_segments.insert(n);

      std::vector<const CanEvent *> new_events;
      new_events.reserve(seg->log->events.size());
      MessageEventsMap msg_events;
      std::set<std::string> cereal_names;
      for (const Event &e : seg->log->events) {
        if (e.which != cereal::Event::Which::CAN && e.which != cereal::Event::Which::SENDCAN) {
          capnp::FlatArrayMessageReader reader(e.data);
          discovery_.observe(reader.getRoot<cereal::Event>());
        }
        if (e.which == cereal::Event::Which::CAN) {
          capnp::FlatArrayMessageReader reader(e.data);
          auto event = reader.getRoot<cereal::Event>();
          for (const auto &c : event.getCan()) {
            const CanEvent *ce = newEvent(e.mono_time, c);
            new_events.push_back(ce);
            msg_events[{.source = ce->src, .address = ce->address}].push_back(ce);
          }
        }
      }
      const auto names = discovery_.names();
      cereal_names.insert(names.begin(), names.end());
      postToMainThreadAndWait([&]() {
        cereal_names_.insert(cereal_names.begin(), cereal_names.end());
        insertEvents(new_events, msg_events);
      });
    }
  }
  // Seeking back into already visited segments still changes the cereal cache window.
  postToMainThread([this]() { cerealChanged(); });
}

bool ReplayStream::loadRoute(const std::string &route, const std::string &data_dir, uint32_t replay_flags, bool auto_source) {
  // Keep cereal events available to charts as well as the existing CAN/video data.
  replay.reset(new Replay(route, {}, {}, nullptr, replay_flags | REPLAY_FLAG_ALL_SERVICES, data_dir, auto_source));
  replay->setSegmentCacheLimit(settings.max_cached_minutes);
  replay->installEventFilter([this](const Event *event) { return eventFilter(event); });

  // replay callbacks arrive on replay threads
  replay->onSeeking = [this](double sec) { postToMainThread([this, sec]() { seeking(sec); }); };
  replay->onSeekedTo = [this](double sec) {
    postToMainThread([this, sec]() { seekedTo(sec); });
    waitForSeekFinshed();
  };
  replay->onQLogLoaded = [this](std::shared_ptr<LogReader> qlog) { postToMainThread([this, qlog]() { thumbnail_log_ = qlog; qLogLoaded(qlog); }); };
  replay->onSegmentsMerged = [this]() { mergeSegments(); };

  bool success = replay->load();
  if (!success) {
    std::string message;
    if (replay->lastRouteError() == RouteLoadError::Unauthorized) {
      auto auth_content = util::read_file(util::getenv("HOME") + "/.comma/auth.json");
      if (auth_content.empty()) {
        message = "Authentication Required. Please run the following command to authenticate:\n\n"
                  "python3 openpilot/tools/lib/auth.py\n\n"
                  "This will grant access to routes from your comma account.";
      } else {
        message = "Access Denied. You do not have permission to access route:\n\n" + route + "\n\n"
                  "This is likely a private route.";
      }
    } else if (replay->lastRouteError() == RouteLoadError::NetworkError) {
      message = "Unable to load the route:\n\n " + route + ".\n\nPlease check your network connection and try again.";
    } else if (replay->lastRouteError() == RouteLoadError::FileNotFound) {
      message = "The specified route could not be found:\n\n " + route + ".\n\nPlease check the route name and try again.";
    } else {
      message = "Failed to load route: '" + route + "'";
    }
    error(message);
  }
  return success;
}

bool ReplayStream::eventFilter(const Event *event) {
  static double prev_update_ts = 0;
  if (event->which == cereal::Event::Which::CAN) {
    double current_sec = toSeconds(event->mono_time);
    capnp::FlatArrayMessageReader reader(event->data);
    auto e = reader.getRoot<cereal::Event>();
    for (const auto &c : e.getCan()) {
      MessageId id = {.source = c.getSrc(), .address = c.getAddress()};
      const auto dat = c.getDat();
      updateEvent(id, current_sec, (const uint8_t*)dat.begin(), dat.size());
    }
  }

  double ts = millis_since_boot();
  if ((ts - prev_update_ts) > (1000.0 / STREAM_UPDATE_FPS)) {
    requestUpdateLastMessages();
    prev_update_ts = ts;
  }
  return true;
}

void ReplayStream::pause(bool pause) {
  replay->pause(pause);
  pause ? paused() : resume();
}


std::vector<std::string> ReplayStream::cerealSignalNames() const {
  return {cereal_names_.begin(), cereal_names_.end()};
}

cabana::CerealReader ReplayStream::cerealReader() const {
  if (!replay) return {};
  auto data = replay->getEventData();
  auto cache_state = cereal_cache_;
  // The closure owns the source snapshot and cache, never ReplayStream or Replay.
  // It is safe for a queued job to finish after the user closes this route.
  return [data, cache_state](const std::string &path, const cabana::CerealCancel &cancelled) {
    auto check_cancel = [&]() {
      if (cancelled && cancelled()) throw std::runtime_error("Cereal request cancelled");
    };
    check_cancel();
    const auto service = path.substr(0, path.find('/'));
    auto schema = capnp::Schema::from<cereal::Event>().asStruct();
    auto field = schema.findFieldByName(service);
    uint16_t which = 65535;
    KJ_IF_MAYBE(found, field) { which = found->getProto().getDiscriminantValue(); }
    if (which == 65535) throw std::invalid_argument("Unknown cereal service: " + service);

    std::lock_guard lock(cache_state->mutex);
    check_cancel();
    for (auto &[_, cache] : cache_state->paths) {
      for (auto it = cache.begin(); it != cache.end();) {
        it = data->segments.count(it->first) ? std::next(it) : cache.erase(it);
      }
    }
    // Segment-window eviction alone does not release previously plotted paths.
    // Bound retained decoded fields even when users browse many signals.
    if (!cache_state->paths.count(path) && cache_state->paths.size() >= 64) {
      cache_state->paths.erase(cache_state->paths.begin());
    }
    auto &cache = cache_state->paths[path];
    cabana::CerealPoints result;
    for (const auto &[number, segment] : data->segments) {
      check_cancel();
      auto &entry = cache[number];
      if (entry.segment.lock() != segment) {
        cabana::CerealPoints points;
        size_t count = 0;
        for (const auto &event : segment->log->events) {
          if (!(++count % 256)) check_cancel();
          if (static_cast<uint16_t>(event.which) != which) continue;
          capnp::FlatArrayMessageReader reader(event.data);
          auto value = cabana::cerealSignalValue(reader.getRoot<cereal::Event>(), path);
          points.push_back({event.mono_time, value.value_or(std::numeric_limits<double>::quiet_NaN()),
                            value ? std::nullopt : cabana::cerealSignalText(reader.getRoot<cereal::Event>(), path)});
        }
        // A cancelled query must never publish a partially decoded segment.
        check_cancel();
        entry.points = std::move(points);
        entry.segment = segment;
      }
      result.insert(result.end(), entry.points.begin(), entry.points.end());
    }
    std::stable_sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
      return a.mono_time < b.mono_time;
    });
    check_cancel();
    return result;
  };
}

cabana::CerealHistoryReader ReplayStream::cerealHistoryReader(bool entire_route) const {
  const auto data = replay->getEventData();
  const auto files = replay->route().segments();
  const auto cache = entire_route ? map_history_cache_ : history_cache_;
  const int end_segment = entire_route ? (files.empty() ? -1 : files.rbegin()->first) : (data->segments.empty() ? -1 : data->segments.rbegin()->first);
  return [data, files, cache, end_segment](const std::set<std::string> &paths, const cabana::CerealCancel &cancelled) {
    const auto check = [&]() { if (cancelled && cancelled()) throw std::runtime_error("History query cancelled"); };
    std::unique_lock guard(cache->mutex);
    check();
    // A route has a finite history; drop signals that are no longer needed.
    for (auto it = cache->paths.begin(); it != cache->paths.end();) {
      it = paths.count(it->first) ? std::next(it) : cache->paths.erase(it);
    }
    for (const auto &[number, source] : files) {
      if (number > end_segment) break;
      check();
      std::map<std::string, cabana::CerealPoints> pending;
      for (const auto &path : paths) if (!cache->paths[path].count(number)) pending[path] = {};
      if (pending.empty()) continue;
      std::unique_ptr<LogReader> loaded;
      const LogReader *log = nullptr;
      const auto current = data->segments.find(number);
      if (current != data->segments.end()) log = current->second->log.get();
      else {
        loaded = std::make_unique<LogReader>();
        const auto url = source.rlog.empty() ? source.qlog : source.rlog;
        if (url.empty()) throw std::runtime_error("Missing history log for segment " + std::to_string(number));
        std::atomic<bool> abort{false}, finished{false};
        std::mutex wake_mutex;
        std::condition_variable wake;
        std::thread monitor([&]() {
          std::unique_lock lock(wake_mutex);
          while (!finished.load()) {
            if (cancelled && cancelled()) abort = true;
            wake.wait_for(lock, std::chrono::milliseconds(20));
          }
        });
        bool success = false;
        try { success = loaded->load(url, &abort, true); }
        catch (...) { finished = true; wake.notify_all(); monitor.join(); throw; }
        finished = true; wake.notify_all(); monitor.join();
        check();
        if (!success) throw std::runtime_error("Cannot load calculation history segment " + std::to_string(number));
        log = loaded.get();
      }
      size_t count = 0;
      for (const auto &event : log->events) {
        if (!(++count % 256)) check();
        capnp::FlatArrayMessageReader reader(event.data);
        auto root = reader.getRoot<cereal::Event>();
        const auto active = capnp::toDynamic(root).which();
        KJ_IF_MAYBE(field, active) {
          const std::string service = field->getProto().getName().cStr();
          for (auto &[path, points] : pending) {
            if (path.substr(0, path.find('/')) != service) continue;
            auto value = cabana::cerealSignalValue(root, path);
            points.push_back({event.mono_time, value.value_or(std::numeric_limits<double>::quiet_NaN()),
                              value ? std::nullopt : cabana::cerealSignalText(root, path)});
          }
        }
      }
      check();
      for (auto &[path, points] : pending) cache->paths[path][number] = std::move(points);
    }
    std::map<std::string, cabana::CerealPoints> result;
    for (const auto &path : paths) {
      auto &points = result[path];
      for (const auto &[number, segment] : cache->paths[path]) {
        if (number <= end_segment) points.insert(points.end(), segment.begin(), segment.end());
      }
      std::stable_sort(points.begin(), points.end(), [](const auto &a, const auto &b) { return a.mono_time < b.mono_time; });
    }
    check();
    return result;
  };
}

cabana::RecordedCameraReader ReplayStream::cameraReader(const std::string &view) const {
  const auto data = replay->getEventData();
  const auto files = replay->route().segments();
  const auto thumbnails = thumbnail_log_;
  auto thumbnail_cache = std::make_shared<cabana::RecordedFrame>();
  return [data, files, view, thumbnails, thumbnail_cache](uint64_t time) -> std::optional<cabana::RecordedFrame> {
    if (view == "thumbnail") {
      const Event *best = nullptr;
      const auto find = [&](const auto &events) {
        auto end = std::upper_bound(events.begin(), events.end(), time,
                                   [](uint64_t requested, const auto &event) { return requested < event.mono_time; });
        while (end != events.begin()) {
          const auto &candidate = *--end;
          if (best && candidate.mono_time <= best->mono_time) break;
          if (candidate.which == cereal::Event::Which::THUMBNAIL) { best = &candidate; break; }
        }
      };
      find(data->events);
      if (thumbnails) find(thumbnails->events);
      if (!best) return std::nullopt;
      if (thumbnail_cache->mono_time == best->mono_time && thumbnail_cache->jpeg) return *thumbnail_cache;
      capnp::FlatArrayMessageReader reader(best->data);
      auto thumbnail = reader.getRoot<cereal::Event>().getThumbnail();
      auto bytes = thumbnail.getThumbnail();
      cabana::RecordedFrame frame;
      frame.mono_time = best->mono_time; frame.path = "thumbnail"; frame.index = thumbnail.getFrameId();
      frame.jpeg = std::make_shared<const std::vector<uint8_t>>(bytes.begin(), bytes.end());
      *thumbnail_cache = frame;
      return frame;
    }
    auto end = std::upper_bound(data->events.begin(), data->events.end(), time,
                               [](uint64_t requested, const auto &event) { return requested < event.mono_time; });
    while (end != data->events.begin()) {
      const auto &event = *--end;
      if (time - event.mono_time > 2000000000ULL) break;
      const bool cabin = view == "cabin" || view == "driver";
      const bool wide = view == "wide_road";
      const bool qroad = event.which == cereal::Event::Which::Q_NARROW_ROAD_ENCODE_IDX;
      const auto wanted = cabin ? cereal::Event::Which::CABIN_ENCODE_IDX : wide ? cereal::Event::Which::WIDE_ROAD_ENCODE_IDX : cereal::Event::Which::NARROW_ROAD_ENCODE_IDX;
      if (event.which != wanted && !(!cabin && !wide && qroad)) continue;
      if (view == "qroad" && !qroad) continue;
      capnp::FlatArrayMessageReader reader(event.data);
      auto root = reader.getRoot<cereal::Event>();
      auto index = cabin ? root.getCabinEncodeIdx() : wide ? root.getWideRoadEncodeIdx() : qroad ? root.getQNarrowRoadEncodeIdx() : root.getNarrowRoadEncodeIdx();
      const int segment = event.eidx_segnum >= 0 ? event.eidx_segnum : index.getSegmentNum();
      const auto source = files.find(segment);
      if (source == files.end()) continue;
      if (qroad && view != "qroad" && !source->second.narrow_road_cam.empty()) continue;
      auto path = cabin ? source->second.cabin_cam : wide ? source->second.wide_road_cam :
                  qroad ? source->second.qcamera : source->second.narrow_road_cam;
      if (path.empty()) return std::nullopt;
      return cabana::RecordedFrame{event.mono_time, path, cabin ? 1 : wide ? 2 : 0, static_cast<int>(index.getSegmentId())};
    }
    return std::nullopt;
  };
}
