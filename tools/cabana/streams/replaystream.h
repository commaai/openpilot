#pragma once

#include <algorithm>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

#include "common/prefix.h"
#include "tools/cabana/streams/abstractstream.h"

class ReplayStream : public AbstractStream {
  Q_OBJECT

public:
  ReplayStream(QObject *parent);
  void start() override;
  bool loadRoute(const QString &route, const QString &data_dir, uint32_t replay_flags = REPLAY_FLAG_NONE);
  bool eventFilter(const Event *event);
  void seekTo(double ts) override { replay->seekTo(std::max(double(0), ts), false); }
  inline QString routeName() const override { return replay->route()->name(); }
  inline QString carFingerprint() const override { return replay->carFingerprint().c_str(); }
  double totalSeconds() const override { return replay->totalSeconds(); }
  inline VisionStreamType visionStreamType() const override { return replay->hasFlag(REPLAY_FLAG_ECAM) ? VISION_STREAM_WIDE_ROAD : VISION_STREAM_ROAD; }
  inline QDateTime beginDateTime() const { return replay->route()->datetime(); }
  inline double routeStartTime() const override { return replay->routeStartTime() / (double)1e9; }
  inline double currentSec() const override { return replay->currentSeconds(); }
  inline const Route *route() const override { return replay->route(); }
  inline void setSpeed(float speed) override { replay->setSpeed(speed); }
  inline float getSpeed() const { return replay->getSpeed(); }
  inline Replay *getReplay() const { return replay.get(); }
  inline bool isPaused() const override { return replay->isPaused(); }
  void pause(bool pause) override;
  inline const std::vector<std::tuple<double, double, TimelineType>> getTimeline() override { return replay->getTimeline(); }
  static AbstractOpenStreamWidget *widget(AbstractStream **stream);

signals:
  void qLogLoaded(int segnum, std::shared_ptr<LogReader> qlog);

private:
  void mergeSegments();
  std::unique_ptr<Replay> replay = nullptr;
  std::set<int> processed_segments;
  std::unique_ptr<OpenpilotPrefix> op_prefix;
};

class OpenReplayWidget : public AbstractOpenStreamWidget {
  Q_OBJECT

public:
  OpenReplayWidget(AbstractStream **stream);
  bool open() override;
  QString title() override { return tr("&Replay"); }

private:
  QLineEdit *route_edit;
  QComboBox *choose_video_cb;
};
