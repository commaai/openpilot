#include "tools/cabana/canmessages.h"
#include "tools/cabana/dbcmanager.h"

CANMessages *can = nullptr;

CANMessages::CANMessages(QObject *parent) : QObject(parent) {
  can = this;
  QObject::connect(this, &CANMessages::received, this, &CANMessages::process, Qt::QueuedConnection);
  QObject::connect(&settings, &Settings::changed, this, &CANMessages::settingChanged);
}

CANMessages::~CANMessages() {
  replay->stop();
}

static bool event_filter(const Event *e, void *opaque) {
  CANMessages *c = (CANMessages *)opaque;
  return c->eventFilter(e);
}

bool CANMessages::loadRoute(const QString &route, const QString &data_dir, uint32_t replay_flags) {
  replay = new Replay(route, {"can", "roadEncodeIdx", "wideRoadEncodeIdx", "carParams"}, {}, nullptr, replay_flags, data_dir, this);
  replay->setSegmentCacheLimit(settings.cached_segment_limit);
  replay->installEventFilter(event_filter, this);
  QObject::connect(replay, &Replay::seekedTo, this, &CANMessages::seekedTo);
  QObject::connect(replay, &Replay::segmentsMerged, this, &CANMessages::eventsMerged);
  QObject::connect(replay, &Replay::streamStarted, this, &CANMessages::streamStarted);
  if (replay->load()) {
    replay->start();
    return true;
  }
  return false;
}

QList<QPointF> CANMessages::findSignalValues(const QString &id, const Signal *signal, double value, FindFlags flag, int max_count) {
  auto evts = events();
  if (!evts) return {};

  QList<QPointF> ret;
  ret.reserve(max_count);
  auto [bus, address] = DBCManager::parseId(id);
  for (auto &evt : *evts) {
    if (evt->which != cereal::Event::Which::CAN) continue;

    for (const auto &c : evt->event.getCan()) {
      if (bus == c.getSrc() && address == c.getAddress()) {
        double val = get_raw_value((uint8_t *)c.getDat().begin(), c.getDat().size(), *signal);
        if ((flag == EQ && val == value) || (flag == LT && val < value) || (flag == GT && val > value)) {
          ret.push_back({(evt->mono_time / (double)1e9) - can->routeStartTime(), val});
          if (ret.size() >= max_count)
            return ret;
        }
      }
    }
  }
  return ret;
}

void CANMessages::process(QHash<QString, CanData> *messages) {
  for (auto it = messages->begin(); it != messages->end(); ++it) {
    can_msgs[it.key()] = it.value();
  }
  emit updated();
  emit msgsReceived(messages);
  delete messages;
  processing = false;
}

bool CANMessages::eventFilter(const Event *event) {
  static std::unique_ptr new_msgs = std::make_unique<QHash<QString, CanData>>();
  static double prev_update_ts = 0;

  if (event->which == cereal::Event::Which::CAN) {
    double current_sec = replay->currentSeconds();
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
      data.dat = QByteArray((char *)c.getDat().begin(), c.getDat().size());
      data.count = ++counters[id];
      if (double delta = (current_sec - counters_begin_sec); delta > 0) {
        data.freq = data.count / delta;
      }
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

void CANMessages::seekTo(double ts) {
  replay->seekTo(std::max(double(0), ts), false);
  counters_begin_sec = 0;
  emit updated();
}

void CANMessages::settingChanged() {
  replay->setSegmentCacheLimit(settings.cached_segment_limit);
}
