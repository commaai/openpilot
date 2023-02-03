#pragma once

#include <atomic>

#include <QColor>
#include <QHash>

#include "tools/cabana/settings.h"
#include "tools/cabana/util.h"
#include "tools/replay/replay.h"

struct CanData {
  double ts = 0.;
  uint8_t src = 0;
  uint32_t address = 0;
  uint32_t count = 0;
  uint32_t freq = 0;
  QByteArray dat;
  QVector<QColor> colors;
  QVector<double> last_change_t;
};

class AbstractStream : public QObject {
  Q_OBJECT

public:
  AbstractStream(QObject *parent, bool is_live_streaming);
  virtual ~AbstractStream() {};
  inline bool liveStreaming() const { return is_live_streaming; }
  virtual void seekTo(double ts) {}
  virtual QString routeName() const = 0;
  virtual QString carFingerprint() const { return ""; }
  virtual double totalSeconds() const { return 0; }
  virtual double routeStartTime() const { return 0; }
  virtual double currentSec() const = 0;
  virtual QDateTime currentDateTime() const { return {}; }
  virtual const CanData &lastMessage(const QString &id) { return can_msgs[id]; }
  virtual VisionStreamType visionStreamType() const { return VISION_STREAM_ROAD; }
  virtual const Route *route() const { return nullptr; }
  virtual const std::vector<Event *> *events() const = 0;
  virtual void setSpeed(float speed) {}
  virtual bool isPaused() const { return false; }
  virtual void pause(bool pause) {}
  virtual const std::vector<std::tuple<int, int, TimelineType>> getTimeline() { return {}; }

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

  bool is_live_streaming = false;
  std::atomic<double> counters_begin_sec = 0;
  std::atomic<bool> processing = false;
  QHash<QString, uint32_t> counters;
};

// A global pointer referring to the unique AbstractStream object
extern AbstractStream *can;
