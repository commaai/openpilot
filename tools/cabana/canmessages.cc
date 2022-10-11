#include "tools/cabana/canmessages.h"

#include <QDebug>

Q_DECLARE_METATYPE(std::vector<CanData>);

CANMessages *can = nullptr;

CANMessages::CANMessages(QObject *parent) : QObject(parent) {
  can = this;

  qRegisterMetaType<std::vector<CanData>>();
  QObject::connect(this, &CANMessages::received, this, &CANMessages::process, Qt::QueuedConnection);
}

CANMessages::~CANMessages() {
  replay->stop();
}

static bool event_filter(const Event *e, void *opaque) {
  CANMessages *c = (CANMessages *)opaque;
  return c->eventFilter(e);
}

bool CANMessages::loadRoute(const QString &route, const QString &data_dir, bool use_qcam) {
  replay = new Replay(route, {"can", "roadEncodeIdx"}, {}, nullptr, use_qcam ? REPLAY_FLAG_QCAMERA : 0, data_dir, this);
  replay->installEventFilter(event_filter, this);
  QObject::connect(replay, &Replay::segmentsMerged, this, &CANMessages::segmentsMerged);
  if (replay->load()) {
    replay->start();
    return true;
  }
  return false;
}

void CANMessages::process(std::vector<CanData> msgs) {
  static double prev_update_ts = 0;

  for (const auto &can_data : msgs) {
    auto &m = can_msgs[can_data.id];
    while (m.size() >= CAN_MSG_LOG_SIZE) {
      m.pop_front();
    }
    m.push_back(can_data);
    ++counters[can_data.id];
  }
  double now = millis_since_boot();
  if ((now - prev_update_ts) > 1000.0 / FPS) {
    prev_update_ts = now;
    emit updated();
  }

  if (current_sec < begin_sec || current_sec > end_sec) {
    // loop replay in selected range.
    seekTo(begin_sec);
  }
}

bool CANMessages::eventFilter(const Event *event) {
  // drop packets when the GUI thread is calling seekTo. to make sure the current_sec is accurate.
  if (!seeking && event->which == cereal::Event::Which::CAN) {
    current_sec = (event->mono_time - replay->routeStartTime()) / (double)1e9;

    auto can_events = event->event.getCan();
    msgs_buf.clear();
    msgs_buf.reserve(can_events.size());

    for (const auto &c : can_events) {
      CanData &data = msgs_buf.emplace_back();
      data.address = c.getAddress();
      data.bus_time = c.getBusTime();
      data.source = c.getSrc();
      data.dat.append((char *)c.getDat().begin(), c.getDat().size());
      data.id = QString("%1:%2").arg(data.source).arg(data.address, 1, 16);
      data.ts = current_sec;
    }
    emit received(msgs_buf);
  }
  return true;
}

void CANMessages::seekTo(double ts) {
  seeking = true;
  replay->seekTo(ts, false);
  seeking = false;
}

void CANMessages::setRange(double min, double max) {
  if (begin_sec != min || end_sec != max) {
    begin_sec = min;
    end_sec = max;
    is_zoomed = begin_sec != event_begin_sec || end_sec != event_end_sec;
    emit rangeChanged(min, max);
  }
}

void CANMessages::segmentsMerged() {
  auto events = replay->events();
  if (!events || events->empty()) return;

  auto it = std::find_if(events->begin(), events->end(), [=](const Event *e) { return e->which == cereal::Event::Which::CAN; });
  event_begin_sec = it == events->end() ? 0 : ((*it)->mono_time - replay->routeStartTime()) / (double)1e9;
  event_end_sec = double(events->back()->mono_time - replay->routeStartTime()) / 1e9;
  if (!is_zoomed) {
    begin_sec = event_begin_sec;
    end_sec = event_end_sec;
  }
  emit eventsMerged();
}

void CANMessages::resetRange() {
  setRange(event_begin_sec, event_end_sec);
}
