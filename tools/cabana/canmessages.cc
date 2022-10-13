#include "tools/cabana/canmessages.h"

#include <QDebug>
#include <QSettings>

Q_DECLARE_METATYPE(std::vector<CanData>);

Settings settings;
CANMessages *can = nullptr;

CANMessages::CANMessages(QObject *parent) : QObject(parent) {
  can = this;

  qRegisterMetaType<std::vector<CanData>>();
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
  replay = new Replay(route, {"can", "roadEncodeIdx"}, {}, nullptr, use_qcam ? REPLAY_FLAG_QCAMERA : 0, data_dir, this);
  replay->setSegmentCacheLimit(settings.cached_segment_limit);
  replay->installEventFilter(event_filter, this);
  QObject::connect(replay, &Replay::segmentsMerged, this, &CANMessages::segmentsMerged);
  if (replay->load()) {
    replay->start();
    return true;
  }
  return false;
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

// Settings

Settings::Settings() {
  load();
}

void Settings::save() {
  QSettings s("settings", QSettings::IniFormat);
  s.setValue("fps", fps);
  s.setValue("log_size", can_msg_log_size);
  s.setValue("cached_segment", cached_segment_limit);
}

void Settings::load() {
  QSettings s("settings", QSettings::IniFormat);
  fps = s.value("fps", 10).toInt();
  can_msg_log_size = s.value("log_size", 100).toInt();
  cached_segment_limit = s.value("cached_segment", 3.).toInt();
}
