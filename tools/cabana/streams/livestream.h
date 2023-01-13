#pragma once

#include "cereal/messaging/messaging.h"
#include "tools/cabana/streams/abstractstream.h"

class LiveStream : public AbstractStream {
  Q_OBJECT

public:
  LiveStream(QObject *parent);
  ~LiveStream();
  inline QString routeName() const override { return "live streaming"; }
  inline QString carFingerprint() const override { return ""; }
  inline double routeStartTime() const override { return start_ts / (double)1e9; }
  inline double currentSec() const override { return (current_ts - start_ts) / (double)1e9; }
  inline const std::vector<Event *> *events() const override { return &can_events; }

protected:
  void streamThread();

#ifdef HAS_MEMORY_RESOURCE
  std::pmr::monotonic_buffer_resource *mbr = nullptr;
  void *pool_buffer = nullptr;
#endif
  std::vector<Event *> can_events;
  std::vector<Message *> messages;
  uint64_t start_ts = 0;
  uint64_t current_ts = 0;
  QThread *stream_thread;
};
