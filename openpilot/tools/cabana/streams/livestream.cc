#include "tools/cabana/streams/livestream.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>

#include "common/timing.h"
#include "common/util.h"
#include "tools/cabana/settings.h"

struct LiveStream::Logger {
  Logger() : start_ts(seconds_since_epoch()), segment_num(-1) {}

  void write(kj::ArrayPtr<capnp::word> data) {
    int n = (seconds_since_epoch() - start_ts) / 60.0;
    if (std::exchange(segment_num, n) != segment_num) {
      const time_t start_time = start_ts;
      std::tm local_time = {};
      localtime_r(&start_time, &local_time);
      std::ostringstream date;
      date << std::put_time(&local_time, "%Y-%m-%d--%H-%M-%S");
      std::string dir = settings.log_path + "/" + date.str() + "--" + std::to_string(n);
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
  begin_date_time = std::chrono::system_clock::now();
  exit_ = false;
  stream_thread = std::thread(&LiveStream::streamThread, this);
  update_thread = std::thread(&LiveStream::updateThread, this);
}

void LiveStream::stop() {
  exit_ = true;
  if (stream_thread.joinable()) stream_thread.join();
  if (update_thread.joinable()) update_thread.join();
}

void LiveStream::updateThread() {
  while (!exit_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000 / STREAM_UPDATE_FPS));
    // coalesce: skip the request if the main thread hasn't processed the previous one yet.
    if (!update_pending_.exchange(true)) {
      requestUpdateLastMessages();
    }
  }
}

// called in streamThread
void LiveStream::handleEvent(kj::ArrayPtr<capnp::word> data) {
  if (logger) {
    logger->write(data);
  }

  capnp::FlatArrayMessageReader reader(data);
  auto event = reader.getRoot<cereal::Event>();
  const uint64_t mono_time = event.getLogMonoTime();
  std::lock_guard lk(lock);
  if (!received_first_ts_) received_first_ts_ = mono_time;
  received_last_ts_ = std::max(received_last_ts_, mono_time);
  if (event.which() == cereal::Event::Which::CAN) {
    for (const auto &c : event.getCan()) {
      received_events_.push_back(newEvent(mono_time, c));
    }
  } else {
    cabana::extractTelemetry(event, received_telemetry_);
  }
}

// called on the main thread via requestUpdateLastMessages()
void LiveStream::updateLastMessages() {
  update_pending_ = false;
  {
    // merge events received from live stream thread.
    std::lock_guard lk(lock);
    if (!begin_event_ts) begin_event_ts = received_first_ts_;
    lastest_event_ts = std::max(lastest_event_ts, received_last_ts_);
    cabana::mergeTelemetry(telemetry, std::move(received_telemetry_));
    received_telemetry_.clear();
    const double cutoff = lastest_event_ts * 1e-9 - settings.max_cached_minutes * 60;
    for (auto &[path, samples] : telemetry) {
      auto first = std::lower_bound(samples.begin(), samples.end(), cutoff, [](const auto &p, double t) { return p.x < t; });
      if (first != samples.begin()) --first;  // retain the boundary sample for nearest-sample equations
      samples.erase(samples.begin(), first);
    }
    mergeEvents(received_events_);
    uint64_t last_received_ts = !received_events_.empty() ? received_events_.back()->mono_time : 0;
    lastest_event_ts = std::max(lastest_event_ts, last_received_ts);
    received_events_.clear();
  }
  if (begin_event_ts) {
    telemetryChanged();
    updateEvents();
  }
}

void LiveStream::updateEvents() {
  static double prev_speed = 1.0;

  if (first_update_ts == 0) {
    first_update_ts = nanos_since_boot();
    first_event_ts = current_event_ts = lastest_event_ts;
  }

  if (paused_ || prev_speed != speed_) {
    prev_speed = speed_;
    first_update_ts = nanos_since_boot();
    first_event_ts = current_event_ts;
    return;
  }

  uint64_t last_ts = post_last_event && speed_ == 1.0
                       ? lastest_event_ts
                       : first_event_ts + (nanos_since_boot() - first_update_ts) * speed_;
  auto first = std::upper_bound(all_events_.cbegin(), all_events_.cend(), current_event_ts, CompareCanEvent());
  auto last = std::upper_bound(first, all_events_.cend(), last_ts, CompareCanEvent());

  for (auto it = first; it != last; ++it) {
    const CanEvent *e = *it;
    MessageId id = {.source = e->src, .address = e->address};
    updateEvent(id, (e->mono_time - begin_event_ts) / 1e9, e->dat, e->size);
    current_event_ts = e->mono_time;
  }
  current_event_ts = std::min(last_ts, lastest_event_ts);
  current_sec_ = (current_event_ts - begin_event_ts) / 1e9;
  AbstractStream::updateLastMessages();
}

void LiveStream::seekTo(double sec) {
  sec = std::max(0.0, sec);
  first_update_ts = nanos_since_boot();
  current_event_ts = first_event_ts = std::min<uint64_t>(sec * 1e9 + begin_event_ts, lastest_event_ts);
  post_last_event = (first_event_ts == lastest_event_ts);
  seekedTo((current_event_ts - begin_event_ts) / 1e9);
}

void LiveStream::pause(bool pause) {
  paused_ = pause;
  pause ? paused() : resume();
}
