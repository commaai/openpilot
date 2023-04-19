#include "tools/cabana/streams/abstractstream.h"
#include <QTimer>

AbstractStream *can = nullptr;

AbstractStream::AbstractStream(QObject *parent, bool is_live_streaming) : is_live_streaming(is_live_streaming), QObject(parent) {
  can = this;
  new_msgs = std::make_unique<QHash<MessageId, CanData>>();
  QObject::connect(this, &AbstractStream::received, this, &AbstractStream::process, Qt::QueuedConnection);
  QObject::connect(this, &AbstractStream::seekedTo, this, &AbstractStream::updateLastMsgsTo);
}

void AbstractStream::process(QHash<MessageId, CanData> *messages) {
  auto prev_src_size = sources.size();
  for (auto it = messages->begin(); it != messages->end(); ++it) {
    const auto &id = it.key();
    last_msgs[id] = it.value();
    sources.insert(id.source);
  }
  if (sources.size() != prev_src_size) {
    emit sourcesUpdated(sources);
  }
  emit updated();
  emit msgsReceived(messages);
  delete messages;
  processing = false;
}

bool AbstractStream::updateEvent(const Event *event) {
  static double prev_update_ts = 0;
  if (event->which == cereal::Event::Which::CAN) {
    double current_sec = event->mono_time / 1e9 - routeStartTime();
    for (const auto &c : event->event.getCan()) {
      MessageId id = {.source = c.getSrc(), .address = c.getAddress()};
      const auto dat = c.getDat();
      all_msgs[id].compute((const char *)dat.begin(), dat.size(), current_sec, getSpeed());
      if (!new_msgs->contains(id)) {
        new_msgs->insert(id, {});
      }
    }
    double ts = millis_since_boot();
    // delay posting CAN message if UI thread is busy
    if ((ts - prev_update_ts) > (1000.0 / settings.fps) && !processing && !new_msgs->isEmpty()) {
      processing = true;
      prev_update_ts = ts;
      for (auto it = new_msgs->begin(); it != new_msgs->end(); ++it) {
        it.value() = all_msgs[it.key()];
      }
      // use pointer to avoid data copy in queued connection.
      emit received(new_msgs.release());
      new_msgs.reset(new QHash<MessageId, CanData>);
      new_msgs->reserve(100);
    }
  }
  return true;
}

const CanData &AbstractStream::lastMessage(const MessageId &id) {
  static CanData empty_data;
  auto it = last_msgs.find(id);
  return it != last_msgs.end() ? it.value() : empty_data;
}

// it is thread safe to update data in updateLastMsgsTo.
// updateEvent will not be called before replayStream::seekedTo return.
void AbstractStream::updateLastMsgsTo(double sec) {
  new_msgs->clear();
  all_msgs.clear();
  last_msgs.clear();

  uint64_t last_ts = (sec + routeStartTime()) * 1e9;
  for (auto &[id, e] : events_) {
    auto it = std::lower_bound(e.crbegin(), e.crend(), last_ts, [](auto e, uint64_t ts) {
      return e->mono_time > ts;
    });
    if (it != e.crend()) {
      double ts = (*it)->mono_time / 1e9 - routeStartTime();
      auto &m = all_msgs[id];
      m.compute((const char *)(*it)->dat, (*it)->size, ts, getSpeed());
      m.count = std::distance(it, e.crend());
      m.freq = m.count / std::max(1.0, ts);
    }
  }
  last_msgs = all_msgs;
  // use a timer to prevent recursive calls
  QTimer::singleShot(0, [this]() {
    emit updated();
    emit msgsReceived(&last_msgs);
  });
}

void AbstractStream::parseEvents(std::unordered_map<MessageId, std::deque<CanEvent *>> &msgs,
                                 std::vector<Event *>::const_iterator first, std::vector<Event *>::const_iterator last) {
  size_t memory_size = 0;
  for (auto it = first; it != last; ++it) {
    if ((*it)->which == cereal::Event::Which::CAN) {
      for (const auto &c : (*it)->event.getCan()) {
        memory_size += sizeof(CanEvent) + sizeof(uint8_t) * c.getDat().size();
      }
    }
  }

  char *ptr = memory_blocks.emplace_back(new char[memory_size]).get();
  uint64_t ts = 0;
  for (auto it = first; it != last; ++it) {
    if ((*it)->which == cereal::Event::Which::CAN) {
      ts = (*it)->mono_time;
      for (const auto &c : (*it)->event.getCan()) {
        auto dat = c.getDat();
        CanEvent *e = (CanEvent *)ptr;
        e->mono_time = ts;
        e->size = dat.size();
        memcpy(e->dat, (uint8_t *)dat.begin(), e->size);
        msgs[{.source = c.getSrc(), .address = c.getAddress()}].push_back(e);
        ptr += sizeof(CanEvent) + sizeof(uint8_t) * e->size;
      }
    }
  }
  last_event_ts = std::max(last_event_ts, ts);
}

void AbstractStream::mergeEvents(std::vector<Event *>::const_iterator first, std::vector<Event *>::const_iterator last, bool append) {
  if (first == last) return;

  if (append) {
    parseEvents(events_, first, last);
  } else {
    std::unordered_map<MessageId, std::deque<CanEvent *>> new_events;
    parseEvents(new_events, first, last);
    for (auto &[id, new_e] : new_events) {
      auto &e = events_[id];
      auto it = std::upper_bound(e.cbegin(), e.cend(), new_e.front());
      e.insert(it, new_e.cbegin(), new_e.cend());
    }
  }
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

void CanData::compute(const char *can_data, const int size, double current_sec, double playback_speed, uint32_t in_freq) {
  ts = current_sec;
  ++count;
  freq = in_freq == 0 ? count / std::max(1.0, current_sec) : in_freq;
  if (dat.size() != size) {
    dat.resize(size);
    bit_change_counts.resize(size);
    colors = QVector(size, QColor(0, 0, 0, 0));
    last_change_t = QVector(size, ts);
  } else {
    bool lighter = settings.theme == DARK_THEME;
    const QColor &cyan = !lighter ? CYAN : CYAN_LIGHTER;
    const QColor &red = !lighter ? RED : RED_LIGHTER;
    const QColor &greyish_blue = !lighter ? GREYISH_BLUE : GREYISH_BLUE_LIGHTER;

    for (int i = 0; i < size; ++i) {
      const uint8_t last = dat[i];
      const uint8_t cur = can_data[i];

      if (last != cur) {
        double delta_t = ts - last_change_t[i];
        if (delta_t * freq > periodic_threshold) {
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
      } else {
        // Fade out
        float alpha_delta = 1.0 / (freq + 1) / (fade_time * playback_speed);
        colors[i].setAlphaF(std::max(0.0, colors[i].alphaF() - alpha_delta));
      }
    }
  }
  memcpy(dat.data(), can_data, size);
}
