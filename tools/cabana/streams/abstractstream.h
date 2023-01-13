#pragma once

#include <atomic>

#include <QColor>
#include <QHash>

#include "opendbc/can/common_dbc.h"
#include "tools/cabana/settings.h"
#include "tools/replay/replay.h"

struct CanData {
  double ts = 0.;
  uint32_t count = 0;
  uint32_t freq = 0;
  QByteArray dat;
};

class AbstractStream : public QObject {
  Q_OBJECT

public:
  AbstractStream(QObject *parent);
  ~AbstractStream();
  virtual void seekTo(double ts) {}

  virtual inline QString routeName() const { return ""; }
  virtual inline QString carFingerprint() const { return ""; }
  virtual inline double totalSeconds() const { return 0; }
  virtual inline double routeStartTime() const { return 0; }
  virtual inline double currentSec() const { return 0; }
  virtual inline const CanData &lastMessage(const QString &id) { return can_msgs[id]; }
  virtual inline VisionStreamType visionStreamType() const { return VISION_STREAM_WIDE_ROAD; }

  virtual inline const Route* route() const { return nullptr; }
  virtual inline const std::vector<Event *> *events() const { return nullptr; }
  virtual inline void setSpeed(float speed) {  }
  virtual  inline bool isPaused() const { return false; }
  virtual void pause(bool pause) {}
  virtual inline const std::vector<std::tuple<int, int, TimelineType>> getTimeline() { return {}; }

signals:
  void paused();
  void resume();
  void seekedTo(double sec);
  void streamStarted();
  void eventsMerged();
  void updated();
  void msgsReceived(const QHash<QString, CanData> *);
  void received(QHash<QString, CanData> *);

public:
  QMap<QString, CanData> can_msgs;

protected:
  void process(QHash<QString, CanData> *);
  bool updateEvent(const Event *event);
  void settingChanged();

  std::atomic<double> counters_begin_sec = 0;
  std::atomic<bool> processing = false;
  QHash<QString, uint32_t> counters;
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
extern AbstractStream *can;
