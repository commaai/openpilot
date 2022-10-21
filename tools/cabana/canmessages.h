#pragma once

#include <atomic>
#include <deque>
#include <map>

#include <QColor>
#include <QHash>
#include <QList>

#include "opendbc/can/common_dbc.h"
#include "tools/cabana/settings.h"
#include "tools/replay/replay.h"

struct CanData {
  double ts;
  uint16_t bus_time;
  QByteArray dat;
};

class CANMessages : public QObject {
  Q_OBJECT

public:
  enum FindFlags{ EQ, LT, GT };
  CANMessages(QObject *parent);
  ~CANMessages();
  bool loadRoute(const QString &route, const QString &data_dir, bool use_qcam);
  void seekTo(double ts);
  void resetRange();
  void setRange(double min, double max);
  QList<QPointF> findSignalValues(const QString&id, const Signal* signal, double value, FindFlags flag, int max_count);
  bool eventFilter(const Event *event);

  inline std::pair<double, double> range() const { return {begin_sec, end_sec}; }
  inline QString route() const { return routeName; }
  inline QString carFingerprint() const { return replay->carFingerprint().c_str(); }
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
  QString routeName;
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

inline QColor hoverColor(const QColor &color) {
  QColor c = color.convertTo(QColor::Hsv);
  c.setHsv(color.hue(), 180, 180);
  return c;
}

// A global pointer referring to the unique CANMessages object
extern CANMessages *can;
