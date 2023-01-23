#include "tools/cabana/streams/livestream.h"

#include <fstream>
#include <QDateTime>
#include <QStandardPaths>

static std::string logFilePath() {
  // TODO: set log root in setting diloag
  std::string path = (QStandardPaths::writableLocation(QStandardPaths::HomeLocation) + "/cabana_live_streams/" +
                      QDateTime::currentDateTime().toString("yyyy-MM-dd--hh-mm-ss") + "--0").toStdString();
  bool ret = util::create_directories(path, 0755);
  assert(ret);
  return path + "/rlog";
}

LiveStream::LiveStream(QObject *parent, QString address, bool logging)
  : zmq_address(address), logging(logging), AbstractStream(parent, true) {
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
  std::unique_ptr<std::ofstream> fs;
  std::unique_ptr<Context> context(Context::create());
  std::string address = zmq_address.isEmpty() ? "127.0.0.1" : zmq_address.toStdString();
  std::unique_ptr<SubSocket> sock(SubSocket::create(context.get(), "can", address));
  assert(sock != NULL);

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
      if (logging) {
        fs.reset(new std::ofstream(logFilePath(), std::ios::binary | std::ios::out));
      }
      emit streamStarted();
    }
    current_ts = evt->mono_time;
    if (current_ts < start_ts) {
      qDebug() << "stream is looping back to old time stamp";
      start_ts = current_ts.load();
    }
    updateEvent(evt);

    if (fs) {
      fs->write((char *)evt->words.begin(), evt->words.size() * sizeof(capnp::word));
    }
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
