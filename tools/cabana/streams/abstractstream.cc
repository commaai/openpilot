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
  for (auto it = messages->begin(); it != messages->end(); ++it) {
    last_msgs[it.key()] = it.value();
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
      CanData &data = (*new_msgs)[id];
      data.ts = current_sec;
      data.dat = QByteArray((char *)c.getDat().begin(), c.getDat().size());
      data.count = ++counters[id];
      data.freq = data.count / std::max(1.0, current_sec);

      auto &tracker = change_trackers[id];
      tracker.compute(data.dat, data.ts, data.freq);
      data.colors = tracker.colors;
      data.last_change_t = tracker.last_change_t;
      data.bit_change_counts = tracker.bit_change_counts;

      if (!sources.contains(id.source)) {
        sources.insert(id.source);
        emit sourcesUpdated(sources);
      }
    }

    double ts = millis_since_boot();
    if ((ts - prev_update_ts) > (1000.0 / settings.fps) && !processing && !new_msgs->isEmpty()) {
      // delay posting CAN message if UI thread is busy
      processing = true;
      prev_update_ts = ts;
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
  change_trackers.clear();
  last_msgs.clear();
  counters.clear();

  CanEvent last_event = {.mono_time = uint64_t((sec + routeStartTime()) * 1e9)};
  for (auto &[id, e] : events_) {
    auto it = std::lower_bound(e.crbegin(), e.crend(), last_event, std::greater<CanEvent>());
    if (it != e.crend()) {
      auto &m = last_msgs[id];
      m.dat = QByteArray((const char *)it->dat, it->size);
      m.ts = it->mono_time / 1e9 - routeStartTime();
      m.count = std::distance(it, e.crend());
      m.freq = m.count / std::max(1.0, m.ts);
      m.last_change_t = QVector<double>(m.dat.size(), m.ts);
      m.colors = QVector<QColor>(m.dat.size(), QColor(0, 0, 0, 0));
      m.bit_change_counts = QVector<std::array<uint32_t, 8>>(m.dat.size());
      counters[id] = m.count;
    }
  }
  QTimer::singleShot(0, [this]() {
    emit updated();
    emit msgsReceived(&last_msgs);
  });
}

void AbstractStream::parseEvents(std::unordered_map<MessageId, std::deque<CanEvent>> &msgs,
                                 std::vector<Event *>::const_iterator first, std::vector<Event *>::const_iterator last) {
  for (; first != last; ++first) {
    if ((*first)->which == cereal::Event::Which::CAN) {
      for (const auto &c : (*first)->event.getCan()) {
        auto dat = c.getDat();
        auto &m = msgs[{.source = c.getSrc(), .address = c.getAddress()}].emplace_back();
        m.size = std::min(dat.size(), std::size(m.dat));
        memcpy(m.dat, (uint8_t *)dat.begin(), m.size);
        m.mono_time = (*first)->mono_time;
      }
      last_event_ts = std::max(last_event_ts, (*first)->mono_time);
    }
  }
}

void AbstractStream::mergeEvents(std::vector<Event *>::const_iterator first, std::vector<Event *>::const_iterator last, bool append) {
  if (first == last) return;

  if (append) {
    parseEvents(events_, first, last);
  } else {
    std::unordered_map<MessageId, std::deque<CanEvent>> new_events;
    parseEvents(new_events, first, last);
    for (auto &[id, new_e] : new_events) {
      auto &e = events_[id];
      auto it = std::upper_bound(e.cbegin(), e.cend(), new_e.front());
      e.insert(it, new_e.cbegin(), new_e.cend());
    }
  }
  emit eventsMerged();
}
