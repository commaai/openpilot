#include "tools/cabana/streams/livestream.h"

#include <algorithm>
#include <ctime>
#include <fstream>
#include <memory>
#include <utility>

#include "common/timing.h"
#include "common/util.h"
#include "tools/cabana/settings.h"

struct LiveStream::Logger {
  Logger() : start_ts(seconds_since_epoch()), segment_num(-1) {}

  void write(kj::ArrayPtr<capnp::word> data) {
    int n = (seconds_since_epoch() - start_ts) / 60.0;
    if (std::exchange(segment_num, n) != segment_num) {
      std::time_t t = (std::time_t)start_ts;
      char ts_buf[32];
      std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%d--%H-%M-%S", std::localtime(&t));
      std::string dir = settings.log_path + "/" + ts_buf + "--" + std::to_string(n);
      util::create_directories(dir, 0755);
      fs.reset(new std::ofstream(dir + "/rlog", std::ios::binary | std::ios::out));
    }

    auto bytes = data.asBytes();
    fs->write((const char*)bytes.begin(), bytes.size());
  }

  std::unique_ptr<std::ofstream> fs;
  int segment_num;
  uint64_t start_ts;
};

LiveStream::LiveStream() {
  if (settings.log_livestream) {
    logger = std::make_unique<Logger>();
  }
}

LiveStream::~LiveStream() {
  stop();
}

void LiveStream::start() {
  stop_requested_ = false;
  stream_thread = std::thread(&LiveStream::streamThread, this);
  begin_date_time = std::chrono::system_clock::now();
}

void LiveStream::stop() {
  if (!stream_thread.joinable()) return;

  stop_requested_ = true;
  stream_thread.join();
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

void LiveStream::update() {
  AbstractStream::update();

  {
    // merge events received from live stream thread.
    std::lock_guard lk(lock);
    mergeEvents(received_events_);
    uint64_t last_received_ts = !received_events_.empty() ? received_events_.back()->mono_time : 0;
    lastest_event_ts = std::max(lastest_event_ts, last_received_ts);
    received_events_.clear();
  }
  if (!all_events_.empty()) {
    begin_event_ts = all_events_.front()->mono_time;
    updateEvents();
  }
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
}

void LiveStream::seekTo(double sec) {
  sec = std::max(0.0, sec);
  first_update_ts = nanos_since_boot();
  current_event_ts = first_event_ts = std::min<uint64_t>(sec * 1e9 + begin_event_ts, lastest_event_ts);
  post_last_event = (first_event_ts == lastest_event_ts);
  double seeked_to = (current_event_ts - begin_event_ts) / 1e9;
  seekedTo(seeked_to);
  updateLastMsgsTo(seeked_to);
}

void LiveStream::pause(bool pause) {
  paused_ = pause;
  pause ? paused() : resume();
}
