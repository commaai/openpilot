#pragma once

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
  void setSpeed(float speed) override { speed_ = speed; }
  double getSpeed() override { return speed_; }
  bool isPaused() const override { return paused_; }
  void pause(bool pause) override;
  void seekTo(double sec) override;

protected:
  virtual void streamThread() = 0;
  void handleEvent(kj::ArrayPtr<capnp::word> event);

private:
  void startUpdateTimer();
  void timerEvent(QTimerEvent *event) override;
  void updateEvents();

  std::mutex lock;
  QThread *stream_thread;
  std::vector<const CanEvent *> received_events_;

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
