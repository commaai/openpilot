#include "tools/cabana/streams/abstractstream.h"

#include <algorithm>

#include <QTimer>

static const int EVENT_NEXT_BUFFER_SIZE = 6 * 1024 * 1024;  // 6MB

AbstractStream *can = nullptr;

StreamNotifier *StreamNotifier::instance() {
  static StreamNotifier notifier;
  return &notifier;
}

AbstractStream::AbstractStream(QObject *parent) : QObject(parent) {
  assert(parent != nullptr);
  new_msgs = std::make_unique<QHash<MessageId, CanData>>();
  event_buffer = std::make_unique<MonotonicBuffer>(EVENT_NEXT_BUFFER_SIZE);

  QObject::connect(this, &AbstractStream::seekedTo, this, &AbstractStream::updateLastMsgsTo);
  QObject::connect(&settings, &Settings::changed, this, &AbstractStream::updateMasks);
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &AbstractStream::updateMasks);
  QObject::connect(dbc(), &DBCManager::maskUpdated, this, &AbstractStream::updateMasks);
  QObject::connect(this, &AbstractStream::streamStarted, [this]() {
    emit StreamNotifier::instance()->changingStream();
    delete can;
    can = this;
    // TODO: add method stop() to class AbstractStream
    QObject::connect(qApp, &QApplication::aboutToQuit, can, []() {
      qDebug() << "stopping stream thread";
      can->pause(true);
    });
    emit StreamNotifier::instance()->streamStarted();
  });
}

void AbstractStream::updateMasks() {
  std::lock_guard lk(mutex);
  masks.clear();
  if (settings.suppress_defined_signals) {
    for (auto s : sources) {
      if (auto f = dbc()->findDBCFile(s)) {
        for (const auto &[address, m] : f->getMessages()) {
          masks[{.source = (uint8_t)s, .address = address}] = m.mask;
        }
      }
    }
  }
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
    updateMasks();
    emit sourcesUpdated(sources);
  }
  emit updated();
  emit msgsReceived(messages, prev_msg_size != last_msgs.size());
  delete messages;
  processing = false;
}

void AbstractStream::updateEvent(const MessageId &id, double sec, const uint8_t *data, uint8_t size) {
  std::lock_guard lk(mutex);
  auto mask_it = masks.find(id);
  std::vector<uint8_t> *mask = mask_it == masks.end() ? nullptr : &mask_it->second;
  all_msgs[id].compute(id, (const char *)data, size, sec, getSpeed(), mask);
  if (!new_msgs->contains(id)) {
    new_msgs->insert(id, {});
  }
}

bool AbstractStream::postEvents() {
  // delay posting CAN message if UI thread is busy
  if (processing == false) {
    processing = true;
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
    auto mask_it = masks.find(id);
    std::vector<uint8_t> *mask = mask_it == masks.end() ? nullptr : &mask_it->second;
    if (it != ev.crend()) {
      double ts = (*it)->mono_time / 1e9 - routeStartTime();
      auto &m = all_msgs[id];
      m.compute(id, (const char *)(*it)->dat, (*it)->size, ts, getSpeed(), mask);
      m.count = std::distance(it, ev.crend());
    }
  }

  // deep copy all_msgs to last_msgs to avoid multi-threading issue.
  last_msgs = all_msgs;
  last_msgs.detach();
  // use a timer to prevent recursive calls
  QTimer::singleShot(0, [this]() {
    emit updated();
    emit msgsReceived(&last_msgs, true);
  });
}

void AbstractStream::mergeEvents(std::vector<Event *>::const_iterator first, std::vector<Event *>::const_iterator last) {
  static MessageEventsMap msg_events;
  static  std::vector<const CanEvent *> new_events;

  std::for_each(msg_events.begin(), msg_events.end(), [](auto &e) { e.second.clear(); });
  new_events.clear();

  for (auto it = first; it != last; ++it) {
    if ((*it)->which == cereal::Event::Which::CAN) {
      uint64_t ts = (*it)->mono_time;
      for (const auto &c : (*it)->event.getCan()) {
        auto dat = c.getDat();
        CanEvent *e = (CanEvent *)event_buffer->allocate(sizeof(CanEvent) + sizeof(uint8_t) * dat.size());
        e->src = c.getSrc();
        e->address = c.getAddress();
        e->mono_time = ts;
        e->size = dat.size();
        memcpy(e->dat, (uint8_t *)dat.begin(), e->size);

        msg_events[{.source = e->src, .address = e->address}].push_back(e);
        new_events.push_back(e);
      }
    }
  }

  if (!new_events.empty()) {
    for (auto &[id, new_e] : msg_events) {
      if (!new_e.empty()) {
        auto &e = events_[id];
        auto pos = std::upper_bound(e.cbegin(), e.cend(), new_e.front()->mono_time, CompareCanEvent());
        e.insert(pos, new_e.cbegin(), new_e.cend());
      }
    }
    auto pos = std::upper_bound(all_events_.cbegin(), all_events_.cend(), new_events.front()->mono_time, CompareCanEvent());
    all_events_.insert(pos, new_events.cbegin(), new_events.cend());
    emit eventsMerged(msg_events);
  }
  lastest_event_ts = all_events_.empty() ? 0 : all_events_.back()->mono_time;
}

// CanData

namespace {

constexpr int periodic_threshold = 10;
constexpr int start_alpha = 128;
constexpr float fade_time = 2.0;
const QColor CYAN = QColor(0, 187, 255, start_alpha);
const QColor RED = QColor(255, 0, 0, start_alpha);
const QColor GREYISH_BLUE = QColor(102, 86, 169, start_alpha / 2);
const QColor CYAN_LIGHTER = QColor(0, 187, 255, start_alpha).lighter(135);
const QColor RED_LIGHTER = QColor(255, 0, 0, start_alpha).lighter(135);
const QColor GREYISH_BLUE_LIGHTER = QColor(102, 86, 169, start_alpha / 2).lighter(135);

inline QColor blend(const QColor &a, const QColor &b) {
  return QColor((a.red() + b.red()) / 2, (a.green() + b.green()) / 2, (a.blue() + b.blue()) / 2, (a.alpha() + b.alpha()) / 2);
}

// Calculate the frequency of the past minute.
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

void CanData::compute(const MessageId &msg_id, const char *can_data, const int size, double current_sec,
                      double playback_speed, const std::vector<uint8_t> *mask, double in_freq) {
  ts = current_sec;
  ++count;

  if (auto sec = seconds_since_boot(); (sec - last_freq_update_ts) >= 1) {
    last_freq_update_ts = sec;
    freq = !in_freq ? calc_freq(msg_id, ts) : in_freq;
  }

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
      const uint8_t mask_byte = (mask && i < mask->size()) ? (~((*mask)[i])) : 0xff;
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
