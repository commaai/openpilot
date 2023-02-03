#include "tools/cabana/streams/abstractstream.h"

AbstractStream *can = nullptr;

AbstractStream::AbstractStream(QObject *parent, bool is_live_streaming) : is_live_streaming(is_live_streaming), QObject(parent) {
  can = this;
  QObject::connect(this, &AbstractStream::received, this, &AbstractStream::process, Qt::QueuedConnection);
}

void AbstractStream::process(QHash<QString, CanData> *messages) {
  for (auto it = messages->begin(); it != messages->end(); ++it) {
    can_msgs[it.key()] = it.value();
  }
  emit updated();
  emit msgsReceived(messages);
  delete messages;
  processing = false;
}

bool AbstractStream::updateEvent(const Event *event) {
  static std::unique_ptr new_msgs = std::make_unique<QHash<QString, CanData>>();
  static QHash<QString, ChangeTracker> change_trackers;
  static double prev_update_ts = 0;

  if (event->which == cereal::Event::Which::CAN) {
    double current_sec = currentSec();
    if (counters_begin_sec == 0 || counters_begin_sec >= current_sec) {
      new_msgs->clear();
      counters.clear();
      counters_begin_sec = current_sec;
    }

    auto can_events = event->event.getCan();
    for (const auto &c : can_events) {
      QString id = QString("%1:%2").arg(c.getSrc()).arg(c.getAddress(), 1, 16);
      CanData &data = (*new_msgs)[id];
      data.ts = current_sec;
      data.src = c.getSrc();
      data.address = c.getAddress();
      data.dat = QByteArray((char *)c.getDat().begin(), c.getDat().size());
      data.count = ++counters[id];
      if (double delta = (current_sec - counters_begin_sec); delta > 0) {
        data.freq = data.count / delta;
      }
      change_trackers[id].compute(data.dat, data.ts, data.freq);
      data.colors = change_trackers[id].colors;
      data.last_change_t = change_trackers[id].last_change_t;
    }

    double ts = millis_since_boot();
    if ((ts - prev_update_ts) > (1000.0 / settings.fps) && !processing && !new_msgs->isEmpty()) {
      // delay posting CAN message if UI thread is busy
      processing = true;
      prev_update_ts = ts;
      // use pointer to avoid data copy in queued connection.
      emit received(new_msgs.release());
      new_msgs.reset(new QHash<QString, CanData>);
      new_msgs->reserve(100);
    }
  }
  return true;
}
