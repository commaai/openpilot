#pragma once

#include <deque>
#include <memory>
#include <vector>

#include <QBasicTimer>

#include "tools/cabana/streams/abstractstream.h"

class LiveStream : public AbstractStream {
  Q_OBJECT

public:
  LiveStream(QObject *parent);
  virtual ~LiveStream();
  void start() override;
  inline QDateTime beginDateTime() const { return begin_date_time; }
  inline double routeStartTime() const override { return begin_event_ts / 1e9; }
  inline double currentSec() const override { return (current_event_ts - begin_event_ts) / 1e9; }
  void setSpeed(float speed) override { speed_ = speed; }
  double getSpeed() override { return speed_; }
  bool isPaused() const override { return paused_; }
  void pause(bool pause) override;
  void seekTo(double sec) override;

protected:
  virtual void streamThread() = 0;
  void handleEvent(const char *data, const size_t size);

private:
  void startUpdateTimer();
  void timerEvent(QTimerEvent *event) override;
  void updateEvents();

  struct Msg {
    Msg(const char *data, const size_t size) {
      event = ::new Event(aligned_buf.align(data, size));
    }
    ~Msg() { ::delete event; }
    Event *event;
    AlignedBuffer aligned_buf;
  };

  std::mutex lock;
  QThread *stream_thread;
  std::vector<Event *> receivedEvents;
  std::deque<Msg> receivedMessages;

  int timer_id;
  QBasicTimer update_timer;

  QDateTime begin_date_time;
  uint64_t begin_event_ts = 0;
  uint64_t current_event_ts = 0;
  uint64_t first_event_ts = 0;
  uint64_t first_update_ts = 0;
  bool post_last_event = true;
  double speed_ = 1;
  bool paused_ = false;

  struct Logger;
  std::unique_ptr<Logger> logger;
};
