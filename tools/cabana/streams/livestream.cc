#include "tools/cabana/streams/livestream.h"

LiveStream::LiveStream(QObject *parent, QString address) : zmq_address(address), AbstractStream(parent, true) {
  if (!zmq_address.isEmpty()) {
    setenv("ZMQ", "1", 1);
  }

  timer = new QTimer(this);
  timer->callOnTimeout(this, &LiveStream::removeExpiredEvents);
  timer->start(3 * 1000);

  stream_thread = new QThread(this);
  QObject::connect(stream_thread, &QThread::started, [=]() { streamThread(); });
  QObject::connect(stream_thread, &QThread::finished, stream_thread, &QThread::deleteLater);
  stream_thread->start();
}

LiveStream::~LiveStream() {
  stream_thread->requestInterruption();
  stream_thread->quit();
  stream_thread->wait();
  for (Event *e : can_events) ::delete e;
  for (auto m : messages) delete m;
}

void LiveStream::streamThread() {
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
    {
      std::lock_guard lk(lock);
      can_events.push_back(evt);
    }

    if (start_ts == 0) {
      start_ts = evt->mono_time;
      emit streamStarted();
    }
    current_ts = evt->mono_time;
    if (current_ts < start_ts) {
      qDebug() << "stream is looping back to old time stamp";
      start_ts = current_ts.load();
    }
    updateEvent(evt);
    // TODO: write stream to log file to replay it with cabana --data_dir flag.
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
  events_vector.clear();
  std::lock_guard lk(lock);
  events_vector.reserve(can_events.size());
  std::copy(can_events.begin(), can_events.end(), std::back_inserter(events_vector));
  return &events_vector;
}
