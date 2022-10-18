#pragma once

#include <atomic>
#include <deque>
#include <map>

#include <QHash>

#include "tools/replay/replay.h"

class Settings : public QObject {
  Q_OBJECT

public:
  Settings();
  void save();
  void load();

  int fps = 10;
  int can_msg_log_size = 100;
  int cached_segment_limit = 3;
  int chart_height = 200;

signals:
  void changed();
};

struct CanData {
  double ts;
  uint16_t bus_time;
  QByteArray dat;
};

class CANMessages : public QObject {
  Q_OBJECT

public:
  CANMessages(QObject *parent);
  ~CANMessages();
  bool loadRoute(const QString &route, const QString &data_dir, bool use_qcam);
  void seekTo(double ts);
  void resetRange();
  void setRange(double min, double max);
  bool eventFilter(const Event *event);

  inline std::pair<double, double> range() const { return {begin_sec, end_sec}; }
  inline double totalSeconds() const { return replay->totalSeconds(); }
  inline double routeStartTime() const { return replay->routeStartTime() / (double)1e9; }
  inline double currentSec() const { return current_sec; }
  inline bool isZoomed() const { return is_zoomed; }
  inline const std::deque<CanData> &messages(const QString &id) { return can_msgs[id]; }
  inline const CanData &lastMessage(const QString &id) { return can_msgs[id].front(); }

  inline const std::vector<Event *> *events() const { return replay->events(); }
  inline void setSpeed(float speed) { replay->setSpeed(speed); }
  inline bool isPaused() const { return replay->isPaused(); }
  inline void pause(bool pause) { replay->pause(pause); }
  inline const std::vector<std::tuple<int, int, TimelineType>> getTimeline() { return replay->getTimeline(); }

signals:
  void eventsMerged();
  void rangeChanged(double min, double max);
  void updated();
  void received(QHash<QString, std::deque<CanData>> *);

public:
  QMap<QString, std::deque<CanData>> can_msgs;
  std::unique_ptr<QHash<QString, std::deque<CanData>>> received_msgs = nullptr;
  QHash<QString, uint32_t> counters;

protected:
  void process(QHash<QString, std::deque<CanData>> *);
  void segmentsMerged();
  void settingChanged();

  std::atomic<double> current_sec = 0.;
  std::atomic<bool> seeking = false;
  double begin_sec = 0;
  double end_sec = 0;
  double event_begin_sec = 0;
  double event_end_sec = 0;
  bool is_zoomed = false;
  Replay *replay = nullptr;
};

inline QString toHex(const QByteArray &dat) {
  return dat.toHex(' ').toUpper();
}
inline char toHex(uint value) {
  return "0123456789ABCDEF"[value & 0xF];
}

inline const QString &getColor(int i) {
  // TODO: add more colors
  static const QString SIGNAL_COLORS[] = {"#9FE2BF", "#40E0D0", "#6495ED", "#CCCCFF", "#FF7F50", "#FFBF00"};
  return SIGNAL_COLORS[i % std::size(SIGNAL_COLORS)];
}

// A global pointer referring to the unique CANMessages object
extern CANMessages *can;
extern Settings settings;
