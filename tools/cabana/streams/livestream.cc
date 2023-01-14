#include "tools/cabana/streams/livestream.h"

LiveStream::LiveStream(QObject *parent, QString address) : zmq_address(address), AbstractStream(parent, true) {
  cache_seconds = settings.cached_segment_limit * 60 * 1e9;
  if (!zmq_address.isEmpty()) {
    setenv("ZMQ", "1", 1);
  }
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
  for (Message *m : messages) delete m;
}

void LiveStream::streamThread() {
  std::unique_ptr<Context> context(Context::create());
  std::string address = zmq_address.isEmpty() ? "127.0.0.1" : zmq_address.toStdString();
  std::unique_ptr<SubSocket> sock(SubSocket::create(context.get(), "can", address));
  assert(sock != NULL);
  sock->setTimeout(100);

  // run as fast as messages come in
  while (!QThread::currentThread()->isInterruptionRequested()) {
    Message *msg = sock->receive(true);
    if (!msg) {
      QThread::msleep(50);
      continue;
    }

    kj::ArrayPtr<capnp::word> words((capnp::word *)msg->getData(), msg->getSize() / sizeof(capnp::word));
    Event *evt = ::new Event(words);

    {
      std::lock_guard lk(lock);
      can_events.push_back(evt);
      messages.push_back(msg);
      if ((evt->mono_time - can_events.front()->mono_time) > cache_seconds) {
        ::delete can_events.front();
        can_events.pop_front();
        delete messages.front();
        messages.pop_front();
      }
    }
    if (start_ts == 0) {
      start_ts = evt->mono_time;
      emit streamStarted();
    }
    current_ts = evt->mono_time;
    updateEvent(evt);
  }
}

const std::vector<Event *> *LiveStream::events() const {
  std::lock_guard lk(lock);
  events_vector.clear();
  events_vector.reserve(can_events.size());
  std::copy(can_events.begin(), can_events.end(), std::back_inserter(events_vector));
  return &events_vector;
}
