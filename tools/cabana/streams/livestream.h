#pragma once

#include "tools/cabana/streams/abstractstream.h"

class LiveStream : public AbstractStream {
  Q_OBJECT

public:
  LiveStream(QObject *parent, QString address = {});
  virtual ~LiveStream();
  inline QString routeName() const override {
    return QString("Live Streaming From %1").arg(zmq_address.isEmpty() ? "127.0.0.1" : zmq_address);
  }
  inline double routeStartTime() const override { return start_ts / (double)1e9; }
  inline double currentSec() const override { return (current_ts - start_ts) / (double)1e9; }
  void setSpeed(float speed) override { speed_ = std::min<float>(1.0, speed); }
  bool isPaused() const override { return pause_; }
  void pause(bool pause) override;

protected:
  virtual void handleEvent(Event *evt);
  virtual void streamThread();
  void process(QHash<MessageId, CanData> *) override;

  struct Msg {
    Msg(Message *m) {
      event = ::new Event(aligned_buf.align(m));
      delete m;
    }
    ~Msg() { ::delete event; }
    Event *event;
    AlignedBuffer aligned_buf;
  };

  mutable std::mutex lock;
  std::vector<Event *> received;
  std::deque<Msg> messages;
  std::atomic<uint64_t> start_ts = 0;
  std::atomic<uint64_t> current_ts = 0;
  std::atomic<float> speed_ = 1;
  std::atomic<bool> pause_ = false;
  uint64_t last_update_ts = 0;

  const QString zmq_address;
  QThread *stream_thread;
};
