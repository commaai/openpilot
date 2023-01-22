#pragma once

#include "tools/cabana/streams/abstractstream.h"

class LiveStream : public AbstractStream {
  Q_OBJECT

public:
  LiveStream(QObject *parent, QString address = {});
  ~LiveStream();
  inline QString routeName() const override {
    return QString("Live Streaming From %1").arg(zmq_address.isEmpty() ? "127.0.0.1" : zmq_address);
  }
  inline double routeStartTime() const override { return start_ts / (double)1e9; }
  inline double currentSec() const override { return (current_ts - start_ts) / (double)1e9; }
  const std::vector<Event *> *events() const override;

protected:
  void streamThread();
  void updateCachedNS() { cache_ns = (settings.max_cached_minutes * 60) * 1e9; }

  mutable std::mutex lock;
  mutable std::vector<Event *> events_vector;
  std::deque<Event *> can_events;
  std::deque<AlignedBuffer *> messages;
  std::atomic<uint64_t> start_ts = 0;
  std::atomic<uint64_t> current_ts = 0;
  std::atomic<uint64_t> cache_ns = 0;
  const QString zmq_address;
  QThread *stream_thread;
};
