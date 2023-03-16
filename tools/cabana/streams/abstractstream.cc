#include "tools/cabana/streams/abstractstream.h"
#include <QTimer>
#include <QtConcurrent>

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
  auto it = can_msgs.find(id);
  return it != can_msgs.end() ? it.value() : empty_data;
}

static QHash<MessageId, CanData> parseEvents(std::vector<Event *>::const_reverse_iterator first,
                                             std::vector<Event *>::const_reverse_iterator last, double route_start_time) {
  QHash<MessageId, CanData> msgs;
  msgs.reserve(500);
  for (auto it = first; it != last; ++it) {
    if ((*it)->which == cereal::Event::Which::CAN) {
      for (const auto &c : (*it)->event.getCan()) {
        auto &m = msgs[{.source = c.getSrc(), .address = c.getAddress()}];
        if (++m.count == 1) {
          m.ts = ((*it)->mono_time / 1e9) - route_start_time;
          m.dat = QByteArray((char *)c.getDat().begin(), c.getDat().size());
          m.colors = QVector<QColor>(m.dat.size(), QColor(0, 0, 0, 0));
          m.last_change_t = QVector<double>(m.dat.size(), m.ts);
          m.bit_change_counts.resize(m.dat.size());
        }
      }
    }
  }
  return msgs;
};

// it is thread safe to update data in updateLastMsgsTo.
// updateEvent will not be called before replayStream::seekedTo return.
void AbstractStream::updateLastMsgsTo(double sec) {
  uint64_t ts = (sec + routeStartTime()) * 1e9;
  const uint64_t delta = std::max(std::ceil(sec / std::thread::hardware_concurrency()), 30.0) * 1e9;
  const auto evs = events();
  auto first = std::upper_bound(evs->crbegin(), evs->crend(), ts, [](uint64_t ts, auto &e) { return ts > e->mono_time; });
  QFutureSynchronizer<QHash<MessageId, CanData>> synchronizer;
  while(first != evs->crend()) {
    ts = (*first)->mono_time > delta ? (*first)->mono_time - delta : 0;
    auto last = std::lower_bound(first, evs->crend(), ts, [](auto &e, uint64_t ts) { return e->mono_time > ts; });
    synchronizer.addFuture(QtConcurrent::run(parseEvents, first, last, routeStartTime()));
    first = last;
  }
  synchronizer.waitForFinished();

  new_msgs->clear();
  change_trackers.clear();
  can_msgs.clear();
  counters.clear();
  for (const auto &f : synchronizer.futures()) {
    auto msgs = f.result();
    for (auto it = msgs.cbegin(); it != msgs.cend(); ++it) {
      counters[it.key()] += it.value().count;
      auto m = can_msgs.find(it.key());
      if (m == can_msgs.end()) {
        m = can_msgs.insert(it.key(), it.value());
      } else {
        m.value().count += it.value().count;
      }
      m.value().freq = m.value().count / std::max(1.0, m.value().ts);
    }
  }
  QTimer::singleShot(0, [this]() {
    emit updated();
    emit msgsReceived(&can_msgs);
  });
}
