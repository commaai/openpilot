#pragma once

#include "tools/cabana/streams/abstractstream.h"

class LiveStream : public AbstractStream {
  Q_OBJECT

public:
  LiveStream(QObject *parent, QString address = {});
  ~LiveStream();
  inline QString routeName() const override { return "Live Streaming Mode"; }
  inline double routeStartTime() const override { return start_ts / (double)1e9; }
  inline double currentSec() const override { return (current_ts - start_ts) / (double)1e9; }
  const std::vector<Event *> *events() const override;

protected:
  void streamThread();

  mutable std::mutex lock;
  std::deque<Event *> can_events;
  mutable std::vector<Event *> events_vector;
  std::deque<Message *> messages;
  std::atomic<uint64_t> start_ts = 0, current_ts = 0;
  std::atomic<int> cache_seconds = 0;
  const QString zmq_address;
  QThread *stream_thread;
};
