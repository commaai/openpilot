#pragma once

#include <atomic>
#include <deque>
#include <mutex>

#include <QColor>
#include <QHash>
#include <QList>

#include "opendbc/can/common_dbc.h"
#include "tools/cabana/settings.h"
#include "tools/replay/replay.h"

struct CanData {
  double ts = 0.;
  uint32_t count = 0;
  uint32_t freq = 0;
  uint16_t bus_time = 0;
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
  QList<QPointF> findSignalValues(const QString&id, const Signal* signal, double value, FindFlags flag, int max_count);
  bool eventFilter(const Event *event);

  inline QString route() const { return replay->route()->name(); }
  inline QString carFingerprint() const { return replay->carFingerprint().c_str(); }
  inline double totalSeconds() const { return replay->totalSeconds(); }
  inline double routeStartTime() const { return replay->routeStartTime() / (double)1e9; }
  inline double currentSec() const { return replay->currentSeconds(); }
  const std::deque<CanData> messages(const QString &id);
  inline const CanData &lastMessage(const QString &id) { return can_msgs[id]; }

  inline const std::vector<Event *> *events() const { return replay->events(); }
  inline void setSpeed(float speed) { replay->setSpeed(speed); }
  inline bool isPaused() const { return replay->isPaused(); }
  inline void pause(bool pause) { replay->pause(pause); }
  inline const std::vector<std::tuple<int, int, TimelineType>> getTimeline() { return replay->getTimeline(); }

signals:
  void streamStarted();
  void eventsMerged();
  void updated();
  void msgsReceived(const QHash<QString, CanData> *);
  void received(QHash<QString, CanData> *);

public:
  QMap<QString, CanData> can_msgs;

protected:
  void process(QHash<QString, CanData> *);
  void settingChanged();

  Replay *replay = nullptr;
  std::mutex lock;
  std::atomic<double> counters_begin_sec = 0;
  std::atomic<bool> processing = false;
  QHash<QString, uint32_t> counters;
  QHash<QString, std::deque<CanData>> received_msgs;
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
