#include "tools/cabana/streams/livestream.h"

LiveStream::LiveStream(QObject *parent) : AbstractStream(parent, true) {
#ifdef HAS_MEMORY_RESOURCE
  const size_t buf_size = sizeof(Event) * 65000;
  pool_buffer = ::operator new(buf_size);
  mbr = new std::pmr::monotonic_buffer_resource(pool_buffer, buf_size);
#endif

  QObject::connect(this, &LiveStream::newEvent, this, &LiveStream::handleNewEvent);
  stream_thread = new QThread(this);
  QObject::connect(stream_thread, &QThread::started, [=]() { streamThread(); });
  QObject::connect(stream_thread, &QThread::finished, stream_thread, &QThread::deleteLater);
  stream_thread->start();
}

LiveStream::~LiveStream() {
  stream_thread->quit();
  stream_thread->wait();
  for (Event *e : can_events) delete e;
  for (Message *m : messages) delete m;

#ifdef HAS_MEMORY_RESOURCE
  delete mbr;
  ::operator delete(pool_buffer);
#endif
}

void LiveStream::handleNewEvent(Event *e) {
  can_events.push_back(e);
}

void LiveStream::streamThread() {
  std::unique_ptr<Context> context(Context::create());
  std::unique_ptr<SubSocket> sock(SubSocket::create(context.get(), "can"));
  assert(sock != NULL);
  sock->setTimeout(100);

  // run as fast as messages come in
  while (!QThread::currentThread()->isInterruptionRequested()) {
    Message *msg = sock->receive();
    if (!msg) {
      if (errno == EINTR) break;
      continue;
    }
    kj::ArrayPtr<capnp::word> words((capnp::word *)msg->getData(), msg->getSize() / sizeof(capnp::word));
#ifdef HAS_MEMORY_RESOURCE
    Event *evt = new (mbr) Event(words);
#else
    Event *evt = new Event(words);
#endif
    messages.push_back(msg);

    if (start_ts == 0) {
      start_ts = evt->mono_time;
      emit streamStarted();
    }
    current_ts = evt->mono_time;
    emit newEvent(evt);
    updateEvent(evt);
  }
}
