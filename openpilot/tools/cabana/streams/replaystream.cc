#include "tools/cabana/streams/replaystream.h"

#include <QMessageBox>

#include "common/timing.h"
#include "common/util.h"

ReplayStream::ReplayStream() {
  unsetenv("ZMQ");
  setenv("COMMA_CACHE", "/tmp/comma_download_cache", 1);

  op_prefix = std::make_unique<OpenpilotPrefix>();

  settings_connection_ = settings.changed.connect([this]() {
    if (replay) replay->setSegmentCacheLimit(settings.max_cached_minutes);
  });
}

ReplayStream::~ReplayStream() {
  cancelWaits();
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
  replay.reset(new Replay(route, {"can", "narrowRoadEncodeIdx", "cabinEncodeIdx", "wideRoadEncodeIdx", "carParams"},
                          {}, nullptr, replay_flags, data_dir, auto_source));
  replay->setSegmentCacheLimit(settings.max_cached_minutes);
  replay->installEventFilter([this](const Event *event) { return eventFilter(event); });

  // replay callbacks arrive on replay threads
  replay->onSeeking = [this](double sec) { postToMainThread([this, sec]() { seeking(sec); }); };
  replay->onSeekedTo = [this](double sec) {
    postToMainThread([this, sec]() { seekedTo(sec); });
    waitForSeekFinshed();
  };
  replay->onQLogLoaded = [this](std::shared_ptr<LogReader> qlog) { postToMainThread([this, qlog]() { qLogLoaded(qlog); }); };
  replay->onSegmentsMerged = [this]() { postToMainThreadAndWait([this]() { mergeSegments(); }); };

  bool success = replay->load();
  if (!success) {
    if (replay->lastRouteError() == RouteLoadError::Unauthorized) {
      auto auth_content = util::read_file(util::getenv("HOME") + "/.comma/auth.json");
      QString message;
      if (auth_content.empty()) {
        message = "Authentication Required. Please run the following command to authenticate:\n\n"
                  "python3 openpilot/tools/lib/auth.py\n\n"
                  "This will grant access to routes from your comma account.";
      } else {
        message = QString("Access Denied. You do not have permission to access route:\n\n%1\n\n"
                          "This is likely a private route.").arg(QString::fromStdString(route));
      }
      QMessageBox::warning(nullptr, "Access Denied", message);
    } else if (replay->lastRouteError() == RouteLoadError::NetworkError) {
      QMessageBox::warning(nullptr, "Network Error",
                          QString("Unable to load the route:\n\n %1.\n\nPlease check your network connection and try again.").arg(QString::fromStdString(route)));
    } else if (replay->lastRouteError() == RouteLoadError::FileNotFound) {
      QMessageBox::warning(nullptr, "Route Not Found",
                           QString("The specified route could not be found:\n\n %1.\n\nPlease check the route name and try again.").arg(QString::fromStdString(route)));
    } else {
      QMessageBox::warning(nullptr, "Route Load Failed", QString("Failed to load route: '%1'").arg(QString::fromStdString(route)));
    }
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
  if ((ts - prev_update_ts) > (1000.0 / settings.fps)) {
    requestUpdateLastMessages();
    prev_update_ts = ts;
  }
  return true;
}

void ReplayStream::pause(bool pause) {
  replay->pause(pause);
  pause ? paused() : resume();
}
