#include "tools/cabana/streams/abstractstream.h"

#include <algorithm>
#include <utility>

#include "common/timing.h"
#include "tools/cabana/settings.h"

static const int EVENT_NEXT_BUFFER_SIZE = 6 * 1024 * 1024;  // 6MB

AbstractStream *can = nullptr;

StreamNotifier *StreamNotifier::instance() {
  static StreamNotifier notifier;
  return &notifier;
}

AbstractStream::AbstractStream(QObject *parent) : QObject(parent) {
  assert(parent != nullptr);
  event_buffer_ = std::make_unique<MonotonicBuffer>(EVENT_NEXT_BUFFER_SIZE);

  QObject::connect(this, &AbstractStream::privateUpdateLastMsgsSignal, this, &AbstractStream::updateLastMessages, Qt::QueuedConnection);
  QObject::connect(this, &AbstractStream::seekedTo, this, &AbstractStream::updateLastMsgsTo);
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &AbstractStream::updateMasks);
  QObject::connect(dbc(), &DBCManager::maskUpdated, this, &AbstractStream::updateMasks);
  QObject::connect(this, &AbstractStream::streamStarted, [this]() {
    emit StreamNotifier::instance()->changingStream();
    delete can;
    can = this;
    emit StreamNotifier::instance()->streamStarted();
  });
}

void AbstractStream::updateMasks() {
  std::lock_guard lk(mutex_);
  masks_.clear();
  if (!settings.suppress_defined_signals)
    return;

  for (const auto s : sources) {
    for (const auto &[address, m] : dbc()->getMessages(s)) {
      masks_[{.source = (uint8_t)s, .address = address}] = m.mask;
    }
  }
  // clear bit change counts
  for (auto &[id, m] : messages_) {
    auto &mask = masks_[id];
    const int size = std::min(mask.size(), m.last_changes.size());
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < 8; ++j) {
        if (((mask[i] >> (7 - j)) & 1) != 0) m.last_changes[i].bit_change_counts[j] = 0;
      }
    }
  }
}

void AbstractStream::suppressDefinedSignals(bool suppress) {
  settings.suppress_defined_signals = suppress;
  updateMasks();
}

size_t AbstractStream::suppressHighlighted() {
  std::lock_guard lk(mutex_);
  size_t cnt = 0;
  const double cur_ts = currentSec();
  for (auto &[_, m] : messages_) {
    for (auto &last_change : m.last_changes) {
      const double dt = cur_ts - last_change.ts;
      if (dt < 2.0) {
        last_change.suppressed = true;
      }
      // clear bit change counts
      last_change.bit_change_counts.fill(0);
      cnt += last_change.suppressed;
    }
  }
  return cnt;
}

void AbstractStream::clearSuppressed() {
  std::lock_guard lk(mutex_);
  for (auto &[_, m] : messages_) {
    std::for_each(m.last_changes.begin(), m.last_changes.end(), [](auto &c) { c.suppressed = false; });
  }
}

void AbstractStream::updateLastMessages() {
  auto prev_src_size = sources.size();
  auto prev_msg_size = last_msgs.size();
  std::set<MessageId> msgs;
  {
    std::lock_guard lk(mutex_);
    for (const auto &id : new_msgs_) {
      const auto &can_data = messages_[id];
      current_sec_ = std::max(current_sec_, can_data.ts);
      last_msgs[id] = can_data;
      sources.insert(id.source);
    }
    msgs = std::move(new_msgs_);
  }

  if (sources.size() != prev_src_size) {
    updateMasks();
    emit sourcesUpdated(sources);
  }
  emit msgsReceived(&msgs, prev_msg_size != last_msgs.size());
}

void AbstractStream::updateEvent(const MessageId &id, double sec, const uint8_t *data, uint8_t size) {
  std::lock_guard lk(mutex_);
  messages_[id].compute(id, data, size, sec, getSpeed(), masks_[id]);
  new_msgs_.insert(id);
}

const std::vector<const CanEvent *> &AbstractStream::events(const MessageId &id) const {
  static std::vector<const CanEvent *> empty_events;
  auto it = events_.find(id);
  return it != events_.end() ? it->second : empty_events;
}

const CanData &AbstractStream::lastMessage(const MessageId &id) {
  static CanData empty_data = {};
  auto it = last_msgs.find(id);
  return it != last_msgs.end() ? it->second : empty_data;
}

// it is thread safe to update data in updateLastMsgsTo.
// updateLastMsgsTo is always called in UI thread.
void AbstractStream::updateLastMsgsTo(double sec) {
  current_sec_ = sec;
  uint64_t last_ts = (sec + routeStartTime()) * 1e9;
  std::unordered_map<MessageId, CanData> msgs;
  msgs.reserve(events_.size());

  for (const auto &[id, ev] : events_) {
    auto it = std::upper_bound(ev.begin(), ev.end(), last_ts, CompareCanEvent());
    if (it != ev.begin()) {
      auto prev = std::prev(it);
      double ts = (*prev)->mono_time / 1e9 - routeStartTime();
      auto &m = msgs[id];
      double freq = 0;
      // Keep suppressed bits.
      if (auto old_m = messages_.find(id); old_m != messages_.end()) {
        freq = old_m->second.freq;
        m.last_changes.reserve(old_m->second.last_changes.size());
        std::transform(old_m->second.last_changes.cbegin(), old_m->second.last_changes.cend(),
                       std::back_inserter(m.last_changes),
                       [](const auto &change) { return CanData::ByteLastChange{.suppressed = change.suppressed}; });
      }
      m.compute(id, (*prev)->dat, (*prev)->size, ts, getSpeed(), {}, freq);
      m.count = std::distance(ev.begin(), prev) + 1;
    }
  }

  new_msgs_.clear();
  messages_ = std::move(msgs);
  bool id_changed = messages_.size() != last_msgs.size() ||
                    std::any_of(messages_.cbegin(), messages_.cend(),
                                [this](const auto &m) { return !last_msgs.count(m.first); });
  last_msgs = messages_;
  emit msgsReceived(nullptr, id_changed);
}

const CanEvent *AbstractStream::newEvent(uint64_t mono_time, const cereal::CanData::Reader &c) {
  auto dat = c.getDat();
  CanEvent *e = (CanEvent *)event_buffer_->allocate(sizeof(CanEvent) + sizeof(uint8_t) * dat.size());
  e->src = c.getSrc();
  e->address = c.getAddress();
  e->mono_time = mono_time;
  e->size = dat.size();
  memcpy(e->dat, (uint8_t *)dat.begin(), e->size);
  return e;
}

void AbstractStream::mergeEvents(const std::vector<const CanEvent *> &events) {
  static MessageEventsMap msg_events;
  std::for_each(msg_events.begin(), msg_events.end(), [](auto &e) { e.second.clear(); });

  // Group events by message ID
  for (auto e : events) {
    msg_events[{.source = e->src, .address = e->address}].push_back(e);
  }

  if (!events.empty()) {
    for (const auto &[id, new_e] : msg_events) {
      if (!new_e.empty()) {
        auto &e = events_[id];
        auto pos = std::upper_bound(e.cbegin(), e.cend(), new_e.front()->mono_time, CompareCanEvent());
        e.insert(pos, new_e.cbegin(), new_e.cend());
      }
    }
    auto pos = std::upper_bound(all_events_.cbegin(), all_events_.cend(), events.front()->mono_time, CompareCanEvent());
    all_events_.insert(pos, events.cbegin(), events.cend());
    emit eventsMerged(msg_events);
  }
  lastest_event_ts = all_events_.empty() ? 0 : all_events_.back()->mono_time;
}

namespace {

enum Color { GREYISH_BLUE, CYAN, RED};
QColor getColor(int c) {
  constexpr int start_alpha = 128;
  static const QColor colors[] = {
      [GREYISH_BLUE] = QColor(102, 86, 169, start_alpha / 2),
      [CYAN] = QColor(0, 187, 255, start_alpha),
      [RED] = QColor(255, 0, 0, start_alpha),
  };
  return settings.theme == LIGHT_THEME ? colors[c] : colors[c].lighter(135);
}

inline QColor blend(const QColor &a, const QColor &b) {
  return QColor((a.red() + b.red()) / 2, (a.green() + b.green()) / 2, (a.blue() + b.blue()) / 2, (a.alpha() + b.alpha()) / 2);
}

// Calculate the frequency from the past one minute data
double calc_freq(const MessageId &msg_id, double current_sec) {
  const auto &events = can->events(msg_id);
  uint64_t cur_mono_time = (can->routeStartTime() + current_sec) * 1e9;
  uint64_t first_mono_time = std::max<int64_t>(0, cur_mono_time - 59 * 1e9);
  auto first = std::lower_bound(events.begin(), events.end(), first_mono_time, CompareCanEvent());
  auto second = std::lower_bound(first, events.end(), cur_mono_time, CompareCanEvent());
  if (first != events.end() && second != events.end()) {
    double duration = ((*second)->mono_time - (*first)->mono_time) / 1e9;
    uint32_t count = std::distance(first, second);
    return count / std::max(1.0, duration);
  }
  return 0;
}

}  // namespace

void CanData::compute(const MessageId &msg_id, const uint8_t *can_data, const int size, double current_sec,
                      double playback_speed, const std::vector<uint8_t> &mask, double in_freq) {
  ts = current_sec;
  ++count;

  if (auto sec = seconds_since_boot(); (sec - last_freq_update_ts) >= 1) {
    last_freq_update_ts = sec;
    freq = !in_freq ? calc_freq(msg_id, ts) : in_freq;
  }

  if (dat.size() != size) {
    dat.resize(size);
    colors.assign(size, QColor(0, 0, 0, 0));
    last_changes.resize(size);
    std::for_each(last_changes.begin(), last_changes.end(), [current_sec](auto &c) { c.ts = current_sec; });
  } else {
    constexpr int periodic_threshold = 10;
    constexpr float fade_time = 2.0;
    const float alpha_delta = 1.0 / (freq + 1) / (fade_time * playback_speed);

    for (int i = 0; i < size; ++i) {
      auto &last_change = last_changes[i];

      uint8_t mask_byte = last_change.suppressed ? 0x00 : 0xFF;
      if (i < mask.size()) mask_byte &= ~(mask[i]);

      const uint8_t last = dat[i] & mask_byte;
      const uint8_t cur = can_data[i] & mask_byte;
      if (last != cur) {
        const int delta = cur - last;
        // Keep track if signal is changing randomly, or mostly moving in the same direction
        last_change.same_delta_counter += std::signbit(delta) == std::signbit(last_change.delta) ? 1 : -4;
        last_change.same_delta_counter = std::clamp(last_change.same_delta_counter, 0, 16);

        const double delta_t = ts - last_change.ts;
        // Mostly moves in the same direction, color based on delta up/down
        if (delta_t * freq > periodic_threshold || last_change.same_delta_counter > 8) {
          // Last change was while ago, choose color based on delta up or down
          colors[i] = getColor(cur > last ? CYAN : RED);
        } else {
          // Periodic changes
          colors[i] = blend(colors[i], getColor(GREYISH_BLUE));
        }

        // Track bit level changes
        const uint8_t diff = (cur ^ last);
        for (int bit = 0; bit < 8; bit++) {
          if (diff & (1u << bit)) {
            ++last_change.bit_change_counts[7 - bit];
          }
        }

        last_change.ts = ts;
        last_change.delta = delta;
      } else {
        // Fade out
        colors[i].setAlphaF(std::max(0.0, colors[i].alphaF() - alpha_delta));
      }
    }
  }
  memcpy(dat.data(), can_data, size);
}
