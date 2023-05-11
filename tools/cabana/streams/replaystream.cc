#include "tools/cabana/streams/replaystream.h"

ReplayStream::ReplayStream(QObject *parent) : AbstractStream(parent, false) {
  QObject::connect(&settings, &Settings::changed, [this]() {
    if (replay) replay->setSegmentCacheLimit(settings.max_cached_minutes);
  });
}

ReplayStream::~ReplayStream() {
  if (replay) replay->stop();
}

static bool event_filter(const Event *e, void *opaque) {
  return ((ReplayStream *)opaque)->eventFilter(e);
}

void ReplayStream::mergeSegments() {
  for (auto &[n, seg] : replay->segments()) {
    if (seg && seg->isLoaded() && !processed_segments.count(n)) {
      const auto &events = seg->log->events;
      bool append = processed_segments.empty() || *processed_segments.rbegin() < n;
      processed_segments.insert(n);
      mergeEvents(events.cbegin(), events.cend(), append);
    }
  }
}

bool ReplayStream::loadRoute(const QString &route, const QString &data_dir, uint32_t replay_flags) {
  replay.reset(new Replay(route, {"can", "roadEncodeIdx", "wideRoadEncodeIdx", "carParams"}, {}, nullptr, replay_flags, data_dir, this));
  replay->setSegmentCacheLimit(settings.max_cached_minutes);
  replay->installEventFilter(event_filter, this);
  QObject::connect(replay.get(), &Replay::seekedTo, this, &AbstractStream::seekedTo);
  QObject::connect(replay.get(), &Replay::streamStarted, this, &AbstractStream::streamStarted);
  QObject::connect(replay.get(), &Replay::segmentsMerged, this, &ReplayStream::mergeSegments);
  if (replay->load()) {
    replay->start();
    return true;
  }
  return false;
}

bool ReplayStream::eventFilter(const Event *event) {
  if (event->which == cereal::Event::Which::CAN) {
    updateEvent(event);
  }
  return true;
}

void ReplayStream::pause(bool pause) {
  replay->pause(pause);
  emit(pause ? paused() : resume());
}
