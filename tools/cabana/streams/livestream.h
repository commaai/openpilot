#pragma once

#include "cereal/messaging/messaging.h"
#include "tools/cabana/streams/abstractstream.h"

class LiveStream : public AbstractStream {
  Q_OBJECT

public:
  LiveStream(QObject *parent);
  ~LiveStream();
  virtual void seekTo(double ts) {}
  bool eventFilter(const Event *event);

  virtual inline QString routeName() const { return "live streaming"; }
  virtual inline QString carFingerprint() const { return ""; }
  virtual inline double totalSeconds() const { return 0; }
  virtual inline double routeStartTime() const { return start_ts / (double)1e9; }
  virtual inline double currentSec() const { return (current_ts - start_ts) / (double)1e9; }
  virtual inline const CanData &lastMessage(const QString &id) { return can_msgs[id]; }
  virtual inline VisionStreamType visionStreamType() const { return VISION_STREAM_WIDE_ROAD; }

  virtual inline const Route* route() const { return nullptr; }
  virtual inline const std::vector<Event *> *events() const { return &can_events; }
  virtual inline void setSpeed(float speed) {  }
  virtual  inline bool isPaused() const { return false; }
  virtual void pause(bool pause) {}
  virtual inline const std::vector<std::tuple<int, int, TimelineType>> getTimeline() { return {}; }

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
