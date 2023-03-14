#include "tools/cabana/streams/livestream.h"

LiveStream::LiveStream(QObject *parent, QString address) : zmq_address(address), AbstractStream(parent, true) {
  timer = new QTimer(this);
  timer->callOnTimeout(this, &LiveStream::removeExpiredEvents);
  timer->start(3 * 1000);

  stream_thread = new QThread(this);
  QObject::connect(stream_thread, &QThread::started, [=]() { streamThread(); });
  QObject::connect(stream_thread, &QThread::finished, stream_thread, &QThread::deleteLater);
  QTimer::singleShot(0, [this]() { stream_thread->start(); });
}

LiveStream::~LiveStream() {
  stream_thread->requestInterruption();
  stream_thread->quit();
  stream_thread->wait();
  for (Event *e : can_events) ::delete e;
  for (auto m : messages) delete m;
}

void LiveStream::streamThread() {
  if (!zmq_address.isEmpty()) {
    setenv("ZMQ", "1", 1);
  }
  std::unique_ptr<Context> context(Context::create());
  std::string address = zmq_address.isEmpty() ? "127.0.0.1" : zmq_address.toStdString();
  std::unique_ptr<SubSocket> sock(SubSocket::create(context.get(), "can", address));
  assert(sock != NULL);
  sock->setTimeout(50);
  // run as fast as messages come in
  while (!QThread::currentThread()->isInterruptionRequested()) {
    Message *msg = sock->receive(true);
    if (!msg) {
      QThread::msleep(50);
      continue;
    }
    AlignedBuffer *buf = messages.emplace_back(new AlignedBuffer());
    Event *evt = ::new Event(buf->align(msg));
    delete msg;

    handleEvent(evt);
    // TODO: write stream to log file to replay it with cabana --data_dir flag.
  }
}

void LiveStream::handleEvent(Event *evt) {
  if (start_ts == 0 || evt->mono_time < start_ts) {
    if (evt->mono_time < start_ts) {
      qDebug() << "stream is looping back to old time stamp";
    }
    start_ts = current_ts = evt->mono_time;
    emit streamStarted();
  }

  std::lock_guard lk(lock);
  can_events.push_back(evt);
  if (!pause_) {
    if (speed_ < 1 && last_update_ts > 0) {
      auto it = std::upper_bound(can_events.cbegin(), can_events.cend(), current_ts, [](uint64_t ts, auto &e) {
        return ts < e->mono_time;
      });
      if (it != can_events.cend()) {
        bool skip = (nanos_since_boot() - last_update_ts) < ((*it)->mono_time - current_ts) / speed_;
        if (skip) return;

        evt = *it;
      }
    }
    current_ts = evt->mono_time;
    last_update_ts = nanos_since_boot();
    updateEvent(evt);
  }
}

void LiveStream::removeExpiredEvents() {
  std::lock_guard lk(lock);
  if (can_events.size() > 0) {
    const uint64_t max_ns = settings.max_cached_minutes * 60 * 1e9;
    const uint64_t last_ns = can_events.back()->mono_time;
    while (!can_events.empty() && (last_ns - can_events.front()->mono_time) > max_ns) {
      ::delete can_events.front();
      delete messages.front();
      can_events.pop_front();
      messages.pop_front();
    }
  }
}

const std::vector<Event *> *LiveStream::events() const {
  std::lock_guard lk(lock);
  if (events_vector.capacity() <= can_events.size()) {
    events_vector.reserve(can_events.size() * 2);
  }
  events_vector.assign(can_events.begin(), can_events.end());
  return &events_vector;
}

void LiveStream::pause(bool pause) {
  pause_ = pause;
  emit paused();
}
