#include "tools/cabana/canmessages.h"

#include <QDebug>
#include <QSettings>

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

bool CANMessages::loadRoute(const QString &route, const QString &data_dir, bool use_qcam) {
  routeName = route;
  replay = new Replay(route, {"can", "roadEncodeIdx", "carParams"}, {}, nullptr, use_qcam ? REPLAY_FLAG_QCAMERA : 0, data_dir, this);
  replay->setSegmentCacheLimit(settings.cached_segment_limit);
  replay->installEventFilter(event_filter, this);
  QObject::connect(replay, &Replay::segmentsMerged, this, &CANMessages::segmentsMerged);
  if (replay->load()) {
    replay->start();
    return true;
  }
  return false;
}

QList<QPointF> CANMessages::findSignalValues(const QString &id, const Signal *signal, double value, FindFlags flag, int max_count) {
  auto evts = events();
  if (!evts) return {};

  auto l = id.split(':');
  int bus = l[0].toInt();
  uint32_t address = l[1].toUInt(nullptr, 16);

  QList<QPointF> ret;
  ret.reserve(max_count);
  for (auto &evt : *evts) {
    if (evt->which != cereal::Event::Which::CAN) continue;

    for (auto c : evt->event.getCan()) {
      if (bus == c.getSrc() && address == c.getAddress()) {
        double val = get_raw_value((uint8_t *)c.getDat().begin(), c.getDat().size(), *signal);
        if ((flag == EQ && val == value) || (flag == LT && val < value) || (flag == GT && val > value)) {
          ret.push_back({(evt->mono_time / (double)1e9) - can->routeStartTime(), val});
        }
        if (ret.size() >= max_count)
          return ret;
      }
    }
  }
  return ret;
}

void CANMessages::process(QHash<QString, std::deque<CanData>> *messages) {
  for (auto it = messages->begin(); it != messages->end(); ++it) {
    ++counters[it.key()];
    auto &msgs = can_msgs[it.key()];
    const auto &new_msgs = it.value();
    if (new_msgs.size() == settings.can_msg_log_size || msgs.empty()) {
      msgs = std::move(new_msgs);
    } else {
      msgs.insert(msgs.begin(), std::make_move_iterator(new_msgs.begin()), std::make_move_iterator(new_msgs.end()));
      while (msgs.size() >= settings.can_msg_log_size) {
        msgs.pop_back();
      }
    }
  }
  delete messages;

  if (current_sec < begin_sec || current_sec > end_sec) {
    // loop replay in selected range.
    seekTo(begin_sec);
  } else {
    emit updated();
  }
}

bool CANMessages::eventFilter(const Event *event) {
  static double prev_update_sec = 0;
  // drop packets when the GUI thread is calling seekTo. to make sure the current_sec is accurate.
  if (!seeking && event->which == cereal::Event::Which::CAN) {
    if (!received_msgs) {
      received_msgs.reset(new QHash<QString, std::deque<CanData>>);
      received_msgs->reserve(1000);
    }

    current_sec = (event->mono_time - replay->routeStartTime()) / (double)1e9;
    auto can_events = event->event.getCan();
    for (const auto &c : can_events) {
      QString id = QString("%1:%2").arg(c.getSrc()).arg(c.getAddress(), 1, 16);
      auto &list = (*received_msgs)[id];
      while (list.size() >= settings.can_msg_log_size) {
        list.pop_back();
      }
      CanData &data = list.emplace_front();
      data.ts = current_sec;
      data.bus_time = c.getBusTime();
      data.dat.append((char *)c.getDat().begin(), c.getDat().size());
    }

    if (current_sec < prev_update_sec || (current_sec - prev_update_sec) > 1.0 / settings.fps) {
      prev_update_sec = current_sec;
      // use pointer to avoid data copy in queued connection.
      emit received(received_msgs.release());
    }
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

void CANMessages::settingChanged() {
  replay->setSegmentCacheLimit(settings.cached_segment_limit);
}
