#include "tools/cabana/streams/livestream.h"

#include <QThread>
#include <algorithm>
#include <fstream>
#include <memory>

#include "common/timing.h"
#include "common/util.h"

struct LiveStream::Logger {
  Logger() : start_ts(seconds_since_epoch()), segment_num(-1) {}

  void write(kj::ArrayPtr<capnp::word> data) {
    int n = (seconds_since_epoch() - start_ts) / 60.0;
    if (std::exchange(segment_num, n) != segment_num) {
      QString dir = QString("%1/%2--%3")
                        .arg(settings.log_path)
                        .arg(QDateTime::fromSecsSinceEpoch(start_ts).toString("yyyy-MM-dd--hh-mm-ss"))
                        .arg(n);
      util::create_directories(dir.toStdString(), 0755);
      fs.reset(new std::ofstream((dir + "/rlog").toStdString(), std::ios::binary | std::ios::out));
    }

    auto bytes = data.asBytes();
    fs->write((const char*)bytes.begin(), bytes.size());
  }

  std::unique_ptr<std::ofstream> fs;
  int segment_num;
  uint64_t start_ts;
};

LiveStream::LiveStream(QObject *parent) : AbstractStream(parent) {
  if (settings.log_livestream) {
    logger = std::make_unique<Logger>();
  }
  stream_thread = new QThread(this);

  QObject::connect(&settings, &Settings::changed, this, &LiveStream::startUpdateTimer);
  QObject::connect(stream_thread, &QThread::started, [=]() { streamThread(); });
  QObject::connect(stream_thread, &QThread::finished, stream_thread, &QThread::deleteLater);
}

LiveStream::~LiveStream() {
  stop();
}

void LiveStream::startUpdateTimer() {
  update_timer.stop();
  update_timer.start(1000.0 / settings.fps, this);
  timer_id = update_timer.timerId();
}

void LiveStream::start() {
  emit streamStarted();
  stream_thread->start();
  startUpdateTimer();
  begin_date_time = QDateTime::currentDateTime();
}

void LiveStream::stop() {
  if (!stream_thread) return;

  update_timer.stop();
  stream_thread->requestInterruption();
  stream_thread->quit();
  stream_thread->wait();
  stream_thread = nullptr;
}

// called in streamThread
void LiveStream::handleEvent(kj::ArrayPtr<capnp::word> data) {
  if (logger) {
    logger->write(data);
  }

  capnp::FlatArrayMessageReader reader(data);
  auto event = reader.getRoot<cereal::Event>();
  if (event.which() == cereal::Event::Which::CAN) {
    const uint64_t mono_time = event.getLogMonoTime();
    std::lock_guard lk(lock);
    for (const auto &c : event.getCan()) {
      received_events_.push_back(newEvent(mono_time, c));
    }
  }
}

void LiveStream::timerEvent(QTimerEvent *event) {
  if (event->timerId() == timer_id) {
    {
      // merge events received from live stream thread.
      std::lock_guard lk(lock);
      mergeEvents(received_events_);
      received_events_.clear();
    }
    if (!all_events_.empty()) {
      begin_event_ts = all_events_.front()->mono_time;
      updateEvents();
      return;
    }
  }
  QObject::timerEvent(event);
}

void LiveStream::updateEvents() {
  static double prev_speed = 1.0;

  if (first_update_ts == 0) {
    first_update_ts = nanos_since_boot();
    first_event_ts = current_event_ts = all_events_.back()->mono_time;
  }

  if (paused_ || prev_speed != speed_) {
    prev_speed = speed_;
    first_update_ts = nanos_since_boot();
    first_event_ts = current_event_ts;
    return;
  }

  uint64_t last_ts = post_last_event && speed_ == 1.0
                       ? all_events_.back()->mono_time
                       : first_event_ts + (nanos_since_boot() - first_update_ts) * speed_;
  auto first = std::upper_bound(all_events_.cbegin(), all_events_.cend(), current_event_ts, CompareCanEvent());
  auto last = std::upper_bound(first, all_events_.cend(), last_ts, CompareCanEvent());

  for (auto it = first; it != last; ++it) {
    const CanEvent *e = *it;
    MessageId id = {.source = e->src, .address = e->address};
    updateEvent(id, (e->mono_time - begin_event_ts) / 1e9, e->dat, e->size);
    current_event_ts = e->mono_time;
  }
  emit privateUpdateLastMsgsSignal();
}

void LiveStream::seekTo(double sec) {
  sec = std::max(0.0, sec);
  first_update_ts = nanos_since_boot();
  current_event_ts = first_event_ts = std::min<uint64_t>(sec * 1e9 + begin_event_ts, lastEventMonoTime());
  post_last_event = (first_event_ts == lastEventMonoTime());
  emit seekedTo((current_event_ts - begin_event_ts) / 1e9);
}

void LiveStream::pause(bool pause) {
  paused_ = pause;
  emit(pause ? paused() : resume());
}
