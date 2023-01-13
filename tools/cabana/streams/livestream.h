#pragma once

#include "cereal/messaging/messaging.h"
#include "tools/cabana/streams/abstractstream.h"

class LiveStream : public AbstractStream {
  Q_OBJECT

public:
  LiveStream(QObject *parent);
  ~LiveStream();
  inline QString routeName() const override { return "Live Streaming Mode"; }
  inline QString carFingerprint() const override { return ""; }
  inline double routeStartTime() const override { return start_ts / (double)1e9; }
  inline double currentSec() const override { return (current_ts - start_ts) / (double)1e9; }
  inline const std::vector<Event *> *events() const override { return &can_events; }

signals:
  void newEvent(Event *e);

protected:
  void streamThread();
  void handleNewEvent(Event *e);

#ifdef HAS_MEMORY_RESOURCE
  std::pmr::monotonic_buffer_resource *mbr = nullptr;
  void *pool_buffer = nullptr;
#endif
  std::vector<Event *> can_events;
  std::vector<Message *> messages;
  std::atomic<uint64_t> start_ts = 0;
  std::atomic<uint64_t> current_ts = 0;
  QThread *stream_thread;
};
