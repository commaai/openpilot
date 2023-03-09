#include "tools/cabana/streams/abstractstream.h"

AbstractStream *can = nullptr;

AbstractStream::AbstractStream(QObject *parent, bool is_live_streaming) : is_live_streaming(is_live_streaming), QObject(parent) {
  can = this;
  new_msgs = std::make_unique<QHash<MessageId, CanData>>();
  QObject::connect(this, &AbstractStream::received, this, &AbstractStream::process, Qt::QueuedConnection);
  QObject::connect(this, &AbstractStream::seekedTo, this, &AbstractStream::updateLastMsgsTo);
}

void AbstractStream::process(QHash<MessageId, CanData> *messages) {
  for (auto it = messages->begin(); it != messages->end(); ++it) {
    can_msgs[it.key()] = it.value();
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
  auto it = can_msgs.find(id);
  return it != can_msgs.end() ? it.value() : empty_data;
}

void AbstractStream::updateLastMsgsTo(double sec) {
  QHash<MessageId, CanData> last_msgs;
  last_msgs.reserve(can_msgs.size());
  double route_start_time = routeStartTime();
  uint64_t last_ts = (sec + route_start_time) * 1e9;
  auto evs = events();
  auto last = std::upper_bound(evs->rbegin(), evs->rend(), last_ts, [](uint64_t ts, auto &e) { return e->mono_time < ts; });
  for (auto it = last; it != evs->rend(); ++it) {
    if ((*it)->which == cereal::Event::Which::CAN) {
      for (const auto &c : (*it)->event.getCan()) {
        auto &m = last_msgs[{.source = c.getSrc(), .address = c.getAddress()}];
        if (++m.count == 1) {
          m.ts = ((*it)->mono_time / 1e9) - route_start_time;
          m.dat = QByteArray((char *)c.getDat().begin(), c.getDat().size());
          m.colors = QVector<QColor>(m.dat.size(), QColor(0, 0, 0, 0));
          m.last_change_t = QVector<double>(m.dat.size(), m.ts);
          m.bit_change_counts.resize(m.dat.size());
        } else {
          m.freq = m.count / std::max(1.0, m.ts);
        }
      }
    }
  }

  // it is thread safe to update data here.
  // updateEvent will not be called before replayStream::seekedTo return.
  new_msgs->clear();
  change_trackers.clear();
  counters.clear();
  can_msgs.clear();
  for (auto it = last_msgs.cbegin(); it != last_msgs.cend(); ++it) {
    can_msgs[it.key()] = it.value();
    counters[it.key()] = it.value().count;
  }
  emit updated();
  emit msgsReceived(&can_msgs);
}
