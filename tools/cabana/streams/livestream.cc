#include "tools/cabana/streams/livestream.h"

LiveStream::LiveStream(QObject *parent) : AbstractStream(parent, true) {
#ifdef HAS_MEMORY_RESOURCE
  const size_t buf_size = sizeof(Event) * 65000;
  pool_buffer = ::operator new(buf_size);
  mbr = new std::pmr::monotonic_buffer_resource(pool_buffer, buf_size);
#endif
  cache_seconds = settings.cached_segment_limit * 60 * 1e9;
  stream_thread = new QThread(this);
  QObject::connect(stream_thread, &QThread::started, [=]() { streamThread(); });
  QObject::connect(stream_thread, &QThread::finished, stream_thread, &QThread::deleteLater);
  stream_thread->start();
}

LiveStream::~LiveStream() {
  stream_thread->requestInterruption();
  stream_thread->quit();
  stream_thread->wait();
  for (Event *e : can_events) delete e;
  for (Message *m : messages) delete m;

#ifdef HAS_MEMORY_RESOURCE
  delete mbr;
  ::operator delete(pool_buffer);
#endif
}

void LiveStream::streamThread() {
  std::unique_ptr<Context> context(Context::create());
  std::unique_ptr<SubSocket> sock(SubSocket::create(context.get(), "can"));
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
#ifdef HAS_MEMORY_RESOURCE
    Event *evt = new (mbr) Event(words);
#else
    Event *evt = new Event(words);
#endif

    {
      std::lock_guard lk(lock);
      can_events.push_back(evt);
      messages.push_back(msg);
      if ((evt->mono_time - can_events.front()->mono_time) > cache_seconds) {
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
