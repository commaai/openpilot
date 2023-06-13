#include "tools/cabana/streams/abstractstream.h"

#include <QTimer>

AbstractStream *can = nullptr;

StreamNotifier *StreamNotifier::instance() {
  static StreamNotifier notifier;
  return &notifier;
}

AbstractStream::AbstractStream(QObject *parent) : new_msgs(new QHash<MessageId, CanData>()), QObject(parent) {
  assert(parent != nullptr);
  QObject::connect(this, &AbstractStream::seekedTo, this, &AbstractStream::updateLastMsgsTo);
  QObject::connect(this, &AbstractStream::streamStarted, [this]() {
    emit StreamNotifier::instance()->changingStream();
    delete can;
    can = this;
    emit StreamNotifier::instance()->streamStarted();
  });
}

void AbstractStream::updateMessages(QHash<MessageId, CanData> *messages) {
  auto prev_src_size = sources.size();
  auto prev_msg_size = last_msgs.size();
  for (auto it = messages->begin(); it != messages->end(); ++it) {
    const auto &id = it.key();
    last_msgs[id] = it.value();
    sources.insert(id.source);
  }
  if (sources.size() != prev_src_size) {
    emit sourcesUpdated(sources);
  }
  emit updated();
  emit msgsReceived(messages, prev_msg_size != last_msgs.size());
  delete messages;
  processing = false;
}

void AbstractStream::updateEvent(const MessageId &id, double sec, const uint8_t *data, uint8_t size) {
  QList<uint8_t> mask = settings.suppress_defined_signals ? dbc()->mask(id) : QList<uint8_t>();
  all_msgs[id].compute((const char *)data, size, sec, getSpeed(), mask);
  if (!new_msgs->contains(id)) {
    new_msgs->insert(id, {});
  }
}

bool AbstractStream::postEvents() {
  // delay posting CAN message if UI thread is busy
  if (processing.exchange(true) == false) {
    for (auto it = new_msgs->begin(); it != new_msgs->end(); ++it) {
      it.value() = all_msgs[it.key()];
    }
    // use pointer to avoid data copy in queued connection.
    QMetaObject::invokeMethod(this, std::bind(&AbstractStream::updateMessages, this, new_msgs.release()), Qt::QueuedConnection);
    new_msgs.reset(new QHash<MessageId, CanData>);
    new_msgs->reserve(100);
    return true;
  }
  return false;
}

const std::vector<const CanEvent *> &AbstractStream::events(const MessageId &id) const {
  static std::vector<const CanEvent *> empty_events;
  auto it = events_.find(id);
  return it != events_.end() ? it->second : empty_events;
}

const CanData &AbstractStream::lastMessage(const MessageId &id) {
  static CanData empty_data = {};
  auto it = last_msgs.find(id);
  return it != last_msgs.end() ? it.value() : empty_data;
}

// it is thread safe to update data in updateLastMsgsTo.
// updateLastMsgsTo is always called in UI thread.
void AbstractStream::updateLastMsgsTo(double sec) {
  new_msgs.reset(new QHash<MessageId, CanData>);
  all_msgs.clear();
  last_msgs.clear();

  uint64_t last_ts = (sec + routeStartTime()) * 1e9;
  for (auto &[id, ev] : events_) {
    auto it = std::lower_bound(ev.crbegin(), ev.crend(), last_ts, [](auto e, uint64_t ts) {
      return e->mono_time > ts;
    });
    QList<uint8_t> mask = settings.suppress_defined_signals ? dbc()->mask(id) : QList<uint8_t>();
    if (it != ev.crend()) {
      double ts = (*it)->mono_time / 1e9 - routeStartTime();
      auto &m = all_msgs[id];
      m.compute((const char *)(*it)->dat, (*it)->size, ts, getSpeed(), mask);
      m.count = std::distance(it, ev.crend());
      m.freq = m.count / std::max(1.0, ts);
    }
  }
  last_msgs = all_msgs;
  // use a timer to prevent recursive calls
  QTimer::singleShot(0, [this]() {
    emit updated();
    emit msgsReceived(&last_msgs, true);
  });
}

void AbstractStream::mergeEvents(std::vector<Event *>::const_iterator first, std::vector<Event *>::const_iterator last) {
  size_t memory_size = 0;
  size_t events_cnt = 0;
  for (auto it = first; it != last; ++it) {
    if ((*it)->which == cereal::Event::Which::CAN) {
      for (const auto &c : (*it)->event.getCan()) {
        memory_size += sizeof(CanEvent) + sizeof(uint8_t) * c.getDat().size();
        ++events_cnt;
      }
    }
  }
  if (memory_size == 0) return;

  char *ptr = memory_blocks.emplace_back(new char[memory_size]).get();
  std::unordered_map<MessageId, std::deque<const CanEvent *>> new_events_map;
  std::vector<const CanEvent *> new_events;
  new_events.reserve(events_cnt);
  for (auto it = first; it != last; ++it) {
    if ((*it)->which == cereal::Event::Which::CAN) {
      uint64_t ts = (*it)->mono_time;
      for (const auto &c : (*it)->event.getCan()) {
        CanEvent *e = (CanEvent *)ptr;
        e->src = c.getSrc();
        e->address = c.getAddress();
        e->mono_time = ts;
        auto dat = c.getDat();
        e->size = dat.size();
        memcpy(e->dat, (uint8_t *)dat.begin(), e->size);

        new_events_map[{.source = e->src, .address = e->address}].push_back(e);
        new_events.push_back(e);
        ptr += sizeof(CanEvent) + sizeof(uint8_t) * e->size;
      }
    }
  }

  bool append = new_events.front()->mono_time > lastest_event_ts;
  for (auto &[id, new_e] : new_events_map) {
    auto &e = events_[id];
    auto pos = append ? e.end() : std::upper_bound(e.cbegin(), e.cend(), new_e.front(), [](const CanEvent *l, const CanEvent *r) {
      return l->mono_time < r->mono_time;
    });
    e.insert(pos, new_e.cbegin(), new_e.cend());
  }

  auto pos = append ? all_events_.end() : std::upper_bound(all_events_.begin(), all_events_.end(), new_events.front(), [](auto l, auto r) {
    return l->mono_time < r->mono_time;
  });
  all_events_.insert(pos, new_events.cbegin(), new_events.cend());

  lastest_event_ts = all_events_.back()->mono_time;
  emit eventsMerged();
}

// CanData

constexpr int periodic_threshold = 10;
constexpr int start_alpha = 128;
constexpr float fade_time = 2.0;
const QColor CYAN = QColor(0, 187, 255, start_alpha);
const QColor RED = QColor(255, 0, 0, start_alpha);
const QColor GREYISH_BLUE = QColor(102, 86, 169, start_alpha / 2);
const QColor CYAN_LIGHTER = QColor(0, 187, 255, start_alpha).lighter(135);
const QColor RED_LIGHTER = QColor(255, 0, 0, start_alpha).lighter(135);
const QColor GREYISH_BLUE_LIGHTER = QColor(102, 86, 169, start_alpha / 2).lighter(135);

static inline QColor blend(const QColor &a, const QColor &b) {
  return QColor((a.red() + b.red()) / 2, (a.green() + b.green()) / 2, (a.blue() + b.blue()) / 2, (a.alpha() + b.alpha()) / 2);
}

void CanData::compute(const char *can_data, const int size, double current_sec, double playback_speed, const QList<uint8_t> &mask, uint32_t in_freq) {
  ts = current_sec;
  ++count;
  const double sec_to_first_event = current_sec - (can->allEvents().front()->mono_time / 1e9 - can->routeStartTime());
  freq = in_freq == 0 ? count / std::max(1.0, sec_to_first_event) : in_freq;
  if (dat.size() != size) {
    dat.resize(size);
    bit_change_counts.resize(size);
    colors = QVector(size, QColor(0, 0, 0, 0));
    last_change_t.assign(size, ts);
    last_delta.resize(size);
    same_delta_counter.resize(size);
  } else {
    bool lighter = settings.theme == DARK_THEME;
    const QColor &cyan = !lighter ? CYAN : CYAN_LIGHTER;
    const QColor &red = !lighter ? RED : RED_LIGHTER;
    const QColor &greyish_blue = !lighter ? GREYISH_BLUE : GREYISH_BLUE_LIGHTER;

    for (int i = 0; i < size; ++i) {
      const uint8_t mask_byte = (i < mask.size()) ? (~mask[i]) : 0xff;
      const uint8_t last = dat[i] & mask_byte;
      const uint8_t cur = can_data[i] & mask_byte;
      const int delta = cur - last;

      if (last != cur) {
        double delta_t = ts - last_change_t[i];

        // Keep track if signal is changing randomly, or mostly moving in the same direction
        if (std::signbit(delta) == std::signbit(last_delta[i])) {
          same_delta_counter[i] = std::min(16, same_delta_counter[i] + 1);
        } else {
          same_delta_counter[i] = std::max(0, same_delta_counter[i] - 4);
        }

        // Mostly moves in the same direction, color based on delta up/down
        if (delta_t * freq > periodic_threshold || same_delta_counter[i] > 8) {
          // Last change was while ago, choose color based on delta up or down
          colors[i] = (cur > last) ? cyan : red;
        } else {
          // Periodic changes
          colors[i] = blend(colors[i], greyish_blue);
        }

        // Track bit level changes
        const uint8_t tmp = (cur ^ last);
        for (int bit = 0; bit < 8; bit++) {
          if (tmp & (1 << bit)) {
            bit_change_counts[i][bit] += 1;
          }
        }

        last_change_t[i] = ts;
        last_delta[i] = delta;
      } else {
        // Fade out
        float alpha_delta = 1.0 / (freq + 1) / (fade_time * playback_speed);
        colors[i].setAlphaF(std::max(0.0, colors[i].alphaF() - alpha_delta));
      }
    }
  }
  memcpy(dat.data(), can_data, size);
}
