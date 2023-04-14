#include "tools/cabana/streams/livestream.h"

#include <QTimer>

LiveStream::LiveStream(QObject *parent, QString address) : zmq_address(address), AbstractStream(parent, true) {
  stream_thread = new QThread(this);
  QObject::connect(stream_thread, &QThread::started, [=]() { streamThread(); });
  QObject::connect(stream_thread, &QThread::finished, stream_thread, &QThread::deleteLater);
  QTimer::singleShot(0, [this]() { stream_thread->start(); });
}

LiveStream::~LiveStream() {
  stream_thread->requestInterruption();
  stream_thread->quit();
  stream_thread->wait();
}

void LiveStream::streamThread() {
  if (!zmq_address.isEmpty()) {
    setenv("ZMQ", "1", 1);
  }

  std::unique_ptr<std::ofstream> fs;
  if (settings.log_livestream) {
    std::string path = (settings.log_path + "/" + QDateTime::currentDateTime().toString("yyyy-MM-dd--hh-mm-ss") + "--0").toStdString();
    util::create_directories(path, 0755);
    fs.reset(new std::ofstream(path + "/rlog" , std::ios::binary | std::ios::out));
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

    if (fs) {
      fs->write(msg->getData(), msg->getSize());
    }
    std::lock_guard lk(lock);
    handleEvent(messages.emplace_back(msg).event);
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

  received.push_back(evt);
  if (!pause_) {
    if (speed_ < 1 && last_update_ts > 0) {
      auto it = std::upper_bound(received.cbegin(), received.cend(), current_ts, [](uint64_t ts, auto &e) {
        return ts < e->mono_time;
      });
      if (it != received.cend()) {
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

void LiveStream::process(QHash<MessageId, CanData> *last_messages) {
  {
    std::lock_guard lk(lock);
    auto first = std::upper_bound(received.cbegin(), received.cend(), last_event_ts, [](uint64_t ts, auto &e) {
      return ts < e->mono_time;
    });
    mergeEvents(first, received.cend(), true);
    if (speed_ == 1) {
      received.clear();
      messages.clear();
    }
  }
  AbstractStream::process(last_messages);
}

void LiveStream::pause(bool pause) {
  pause_ = pause;
  emit paused();
}
