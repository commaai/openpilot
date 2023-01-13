#pragma once

#include <atomic>

#include <QColor>
#include <QHash>

#include "opendbc/can/common_dbc.h"
#include "tools/cabana/streams/abstractstream.h"
#include "tools/cabana/settings.h"
#include "tools/replay/replay.h"

class ReplayStream : public AbstractStream {
  Q_OBJECT

public:
  ReplayStream(QObject *parent);
  ~ReplayStream();
  bool loadRoute(const QString &route, const QString &data_dir, uint32_t replay_flags = REPLAY_FLAG_NONE);
  void seekTo(double ts);
  bool eventFilter(const Event *event);
  inline QString routeName() const { return replay->route()->name(); }
  inline QString carFingerprint() const { return replay->carFingerprint().c_str(); }
  inline VisionStreamType visionStreamType() const { return replay->hasFlag(REPLAY_FLAG_ECAM) ? VISION_STREAM_WIDE_ROAD : VISION_STREAM_ROAD; }
  inline double totalSeconds() const { return replay->totalSeconds(); }
  inline double routeStartTime() const { return replay->routeStartTime() / (double)1e9; }
  inline double currentSec() const { return replay->currentSeconds(); }
  inline const CanData &lastMessage(const QString &id) { return can_msgs[id]; }

  inline const Route* route() const { return replay->route(); }
  inline const std::vector<Event *> *events() const { return replay->events(); }
  inline void setSpeed(float speed) { replay->setSpeed(speed); }
  inline bool isPaused() const { return replay->isPaused(); }
  void pause(bool pause);
  inline const std::vector<std::tuple<int, int, TimelineType>> getTimeline() { return replay->getTimeline(); }

protected:
  void settingChanged();

  Replay *replay = nullptr;
};
