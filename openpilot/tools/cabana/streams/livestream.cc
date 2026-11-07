#include "tools/cabana/streams/livestream.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <limits>
#include <cstring>
#include <stdexcept>
#include <cstdio>
#include <cerrno>
#include <unistd.h>

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

struct LiveStream::HistorySpool {
  FILE *file = std::tmpfile();
  std::mutex mutex;
  uint64_t end = 0;
  std::string error;
  HistorySpool() { if (!file) throw std::runtime_error("Cannot create live calculation history spool"); }
  ~HistorySpool() { std::fclose(file); }
};

LiveStream::LiveStream() : history_(std::make_shared<HistorySpool>()) {
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
  {
    std::lock_guard guard(cereal_mutex_);
    if (event.which() == cereal::Event::Which::THUMBNAIL) {
      const auto bytes = event.getThumbnail().getThumbnail();
      cabana::RecordedFrame frame;
      frame.mono_time = mono_time; frame.path = "thumbnail";
      frame.jpeg = std::make_shared<const std::vector<uint8_t>>(bytes.begin(), bytes.end());
      const auto position = std::upper_bound(thumbnails_.begin(), thumbnails_.end(), frame.mono_time,
        [](uint64_t time, const auto &item) { return time < item.mono_time; });
      thumbnails_.insert(position, std::move(frame));
      while (!thumbnails_.empty() && thumbnails_.back().mono_time - thumbnails_.front().mono_time > 600000000000ULL) thumbnails_.pop_front();
    }
    first_received_ts_ = first_received_ts_ ? std::min(first_received_ts_, mono_time) : mono_time;
    latest_received_ts_ = std::max(latest_received_ts_, mono_time);
    if (event.which() != cereal::Event::Which::CAN && event.which() != cereal::Event::Which::SENDCAN) {
      {
        std::lock_guard history_lock(history_->mutex);
        const uint64_t bytes = data.size() * sizeof(capnp::word);
        if (history_->error.empty()) {
          if (std::fwrite(&bytes, sizeof(bytes), 1, history_->file) != 1 ||
              std::fwrite(data.begin(), 1, bytes, history_->file) != bytes) {
            history_->error = "Cannot write live calculation history (check free disk space)";
          } else history_->end += sizeof(bytes) + bytes;
        }
      }
      cereal_discovery_.observe(event);
      const auto names = capnp::toDynamic(event).which();
      KJ_IF_MAYBE(field, names) {
        auto entry = std::make_shared<CerealEvent>();
        entry->time = mono_time;
        entry->service = field->getProto().getName().cStr();
        entry->words.resize(data.size());
        memcpy(entry->words.data(), data.begin(), data.size() * sizeof(capnp::word));
        auto position = std::upper_bound(cereal_events_.begin(), cereal_events_.end(), mono_time,
          [](uint64_t time, const auto &item) { return time < item->time; });
        cereal_events_.insert(position, entry);
        cereal_bytes_ += data.size() * sizeof(capnp::word);
        // Bound both age and bytes. Reader snapshots retain their own shared ownership.
        constexpr size_t max_bytes = 256ULL * 1024 * 1024;
        constexpr uint64_t max_age = 600ULL * 1000000000;
        while (!cereal_events_.empty() && (cereal_bytes_ > max_bytes ||
               latest_received_ts_ - cereal_events_.front()->time > max_age)) {
          cereal_bytes_ -= cereal_events_.front()->words.size() * sizeof(capnp::word);
          cereal_events_.pop_front();
        }
        cereal_dirty_ = true;
      }
    }
  }
  if (event.which() == cereal::Event::Which::CAN) {
    std::lock_guard lk(lock);
    for (const auto &c : event.getCan()) {
      received_events_.push_back(newEvent(mono_time, c));
    }
  }
}

// called on the main thread via requestUpdateLastMessages()
void LiveStream::updateLastMessages() {
  update_pending_ = false;
  bool changed;
  {
    std::lock_guard guard(cereal_mutex_);
    if (!begin_event_ts) begin_event_ts = first_received_ts_;
    lastest_event_ts = std::max(lastest_event_ts, latest_received_ts_);
    changed = std::exchange(cereal_dirty_, false);
  }
  {
    // Establish the clock before chart callbacks run during mergeEvents.
    std::lock_guard guard(lock);
    received_events_.erase(std::remove_if(received_events_.begin(), received_events_.end(),
      [this](const auto *event) { return event->mono_time < begin_event_ts; }), received_events_.end());
    std::stable_sort(received_events_.begin(), received_events_.end(),
                     [](const auto *a, const auto *b) { return a->mono_time < b->mono_time; });
    mergeEvents(received_events_);
    received_events_.clear();
  }
  if (begin_event_ts) updateEvents();
  if (changed) cerealChanged();
}

void LiveStream::updateEvents() {

  if (first_update_ts == 0) {
    first_update_ts = nanos_since_boot();
    first_event_ts = current_event_ts = lastest_event_ts;
  }

  if (paused_ || previous_speed_ != speed_) {
    previous_speed_ = speed_;
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
  current_sec_ = toSeconds(current_event_ts);
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

std::vector<std::string> LiveStream::cerealSignalNames() const {
  std::lock_guard guard(cereal_mutex_);
  return cereal_discovery_.names();
}

cabana::CerealReader LiveStream::cerealReader() const {
  std::deque<std::shared_ptr<const CerealEvent>> snapshot;
  {
    std::lock_guard guard(cereal_mutex_);
    snapshot = cereal_events_;
  }
  return [snapshot = std::move(snapshot), origin = begin_event_ts](const std::string &path, const cabana::CerealCancel &cancelled) {
    cabana::CerealPoints result;
    const auto service = path.substr(0, path.find('/'));
    size_t count = 0;
    for (const auto &entry : snapshot) {
      if (!(count++ % 256) && cancelled && cancelled()) throw std::runtime_error("Live cereal request cancelled");
      if (entry->service != service || entry->time < origin) continue;
      capnp::FlatArrayMessageReader reader(kj::arrayPtr(entry->words.data(), entry->words.size()));
      auto value = cabana::cerealSignalValue(reader.getRoot<cereal::Event>(), path);
      result.push_back({entry->time, value.value_or(std::numeric_limits<double>::quiet_NaN()),
                        value ? std::nullopt : cabana::cerealSignalText(reader.getRoot<cereal::Event>(), path)});
    }
    return result;
  };
}

cabana::CerealHistoryReader LiveStream::cerealHistoryReader(bool) const {
  const auto history = history_;
  uint64_t end;
  std::string failure;
  { std::lock_guard guard(history->mutex);
    if (std::fflush(history->file)) history->error = "Cannot flush live calculation history";
    end = history->end; failure = history->error;
  }
  const auto origin = begin_event_ts;
  return [history, end, origin, failure](const std::set<std::string> &paths, const cabana::CerealCancel &cancelled) {
    if (!failure.empty()) throw std::runtime_error(failure);
    std::map<std::string, cabana::CerealPoints> result;
    for (const auto &path : paths) result[path] = {};
    const auto read = [&](void *buffer, size_t bytes, uint64_t offset) {
      auto *target = static_cast<char *>(buffer);
      while (bytes) {
        const auto n = pread(fileno(history->file), target, bytes, offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) throw std::runtime_error("Cannot read live calculation history");
        target += n; bytes -= n; offset += n;
      }
    };
    uint64_t position = 0;
    while (position < end) {
      if (cancelled && cancelled()) throw std::runtime_error("Live history query cancelled");
      uint64_t bytes;
      read(&bytes, sizeof(bytes), position); position += sizeof(bytes);
      if (bytes > end - position || bytes % sizeof(capnp::word)) throw std::runtime_error("Invalid live history frame");
      std::vector<capnp::word> words(bytes / sizeof(capnp::word));
      read(words.data(), bytes, position); position += bytes;
      capnp::FlatArrayMessageReader reader(kj::arrayPtr(words.data(), words.size()));
      auto root = reader.getRoot<cereal::Event>();
      if (root.getLogMonoTime() < origin) continue;
      const auto active = capnp::toDynamic(root).which();
      KJ_IF_MAYBE(field, active) {
        const std::string service = field->getProto().getName().cStr();
        for (auto &[path, points] : result) {
          if (path.substr(0, path.find('/')) != service) continue;
          auto value = cabana::cerealSignalValue(root, path);
          points.push_back({root.getLogMonoTime(), value.value_or(std::numeric_limits<double>::quiet_NaN()),
                            value ? std::nullopt : cabana::cerealSignalText(root, path)});
        }
      }
    }
    for (auto &[_, points] : result) std::stable_sort(points.begin(), points.end(), [](const auto &a, const auto &b) { return a.mono_time < b.mono_time; });
    return result;
  };
}

cabana::RecordedCameraReader LiveStream::cameraReader(const std::string &view) const {
  if (view != "thumbnail") return {};
  std::lock_guard guard(cereal_mutex_);
  return [frames = thumbnails_](uint64_t time) -> std::optional<cabana::RecordedFrame> {
    const auto position = std::upper_bound(frames.begin(), frames.end(), time,
      [](uint64_t requested, const auto &frame) { return requested < frame.mono_time; });
    return position == frames.begin() ? std::nullopt : std::optional<cabana::RecordedFrame>(*std::prev(position));
  };
}
