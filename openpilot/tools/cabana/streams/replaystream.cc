#include "tools/cabana/streams/replaystream.h"

#include <cstdio>

#include "common/util.h"
#include "tools/cabana/settings.h"

ReplayStream::ReplayStream() {
  unsetenv("ZMQ");
  setenv("COMMA_CACHE", "/tmp/comma_download_cache", 1);

  // TODO: Remove when OpenpilotPrefix supports ZMQ
#ifndef __APPLE__
  op_prefix = std::make_unique<OpenpilotPrefix>();
#endif

  settings.changed.connect([this]() {
    if (replay) replay->setSegmentCacheLimit(settings.max_cached_minutes);
  });
}

ReplayStream::~ReplayStream() {
  {
    std::lock_guard lk(merge_mutex_);
    aborting_ = true;
  }
  merge_cv_.notify_all();
  replay.reset();
}

void ReplayStream::mergeSegments() {
  auto event_data = replay->getEventData();
  for (const auto &[n, seg] : event_data->segments) {
    if (!processed_segments.count(n)) {
      processed_segments.insert(n);

      std::vector<const CanEvent *> new_events;
      new_events.reserve(seg->log->events.size());
      for (const Event &e : seg->log->events) {
        if (e.which == cereal::Event::Which::CAN) {
          capnp::FlatArrayMessageReader reader(e.data);
          auto event = reader.getRoot<cereal::Event>();
          for (const auto &c : event.getCan()) {
            new_events.push_back(newEvent(e.mono_time, c));
          }
        }
      }
      mergeEvents(new_events);
    }
  }
}

bool ReplayStream::loadRoute(const std::string &route, const std::string &data_dir, uint32_t replay_flags, bool auto_source) {
  replay.reset(new Replay(route, {"can", "roadEncodeIdx", "driverEncodeIdx", "wideRoadEncodeIdx", "carParams"},
                          {}, nullptr, replay_flags, data_dir, auto_source));
  replay->setSegmentCacheLimit(settings.max_cached_minutes);
  replay->installEventFilter([this](const Event *event) { return eventFilter(event); });

  // Replay invokes these callbacks either synchronously on the calling
  // thread (e.g. seeking into an already-loaded segment resolves inline) or
  // asynchronously from a replay-owned thread. The old Qt::AutoConnection
  // handled this transparently (direct call vs. queued); onUiThread() makes
  // the same distinction explicit: run inline on the UI thread, otherwise
  // enqueue for AbstractStream::update() to drain -- never touch UI state
  // directly from a replay thread. See abstractstream.h for the contract.
  replay->onSeeking = [this](double sec) {
    auto action = [this, sec]() {
      current_sec_ = sec;
      seeking(sec);
    };
    if (onUiThread()) action(); else enqueue(std::move(action));
  };
  replay->onSeekedTo = [this](double sec) {
    auto action = [this, sec]() {
      seekedTo(sec);
      updateLastMsgsTo(sec);
    };
    if (onUiThread()) {
      action();
    } else {
      enqueue(std::move(action));
    }
    // Always consume seek_finished_ (updateLastMsgsTo sets it unconditionally);
    // inline it returns immediately, async it blocks the replay thread until
    // the UI thread has drained the action.
    waitForSeekFinshed();
  };
  replay->onQLogLoaded = [this](std::shared_ptr<LogReader> qlog) {
    auto action = [this, qlog]() { qLogLoaded(qlog); };
    if (onUiThread()) action(); else enqueue(std::move(action));
  };
  replay->onSegmentsMerged = [this]() {
    if (onUiThread()) {
      mergeSegments();
      return;
    }
    // Mirrors the old Qt::BlockingQueuedConnection: block this (replay)
    // thread until the UI thread has run mergeSegments().
    std::unique_lock lock(merge_mutex_);
    if (aborting_) return;
    merge_done_ = false;
    enqueue([this]() {
      mergeSegments();
      std::lock_guard lk(merge_mutex_);
      merge_done_ = true;
      merge_cv_.notify_one();
    });
    merge_cv_.wait(lock, [this]() { return merge_done_ || aborting_.load(); });
  };

  bool success = replay->load();
  if (!success) {
    if (replay->lastRouteError() == RouteLoadError::Unauthorized) {
      auto auth_content = util::read_file(util::getenv("HOME") + "/.comma/auth.json");
      if (auth_content.empty()) {
        fprintf(stderr, "Authentication Required. Please run the following command to authenticate:\n\n"
                        "python3 openpilot/tools/lib/auth.py\n\n"
                        "This will grant access to routes from your comma account.\n");
      } else {
        fprintf(stderr, "Access Denied. You do not have permission to access route:\n\n%s\n\n"
                        "This is likely a private route.\n", route.c_str());
      }
    } else if (replay->lastRouteError() == RouteLoadError::NetworkError) {
      fprintf(stderr, "Unable to load the route:\n\n %s.\n\nPlease check your network connection and try again.\n", route.c_str());
    } else if (replay->lastRouteError() == RouteLoadError::FileNotFound) {
      fprintf(stderr, "The specified route could not be found:\n\n %s.\n\nPlease check the route name and try again.\n", route.c_str());
    } else {
      fprintf(stderr, "Failed to load route: '%s'\n", route.c_str());
    }
  }
  return success;
}

bool ReplayStream::eventFilter(const Event *event) {
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
  return true;
}

void ReplayStream::pause(bool pause) {
  replay->pause(pause);
  pause ? paused() : resume();
}
