#pragma once

#include <array>
#include <atomic>
#include <deque>
#include <unordered_map>
#include <QColor>
#include <QHash>

#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/util.h"
#include "tools/replay/replay.h"

struct CanData {
  void compute(const char *dat, const int size, double current_sec, double playback_speed, uint32_t in_freq = 0);

  double ts = 0.;
  uint32_t count = 0;
  double freq = 0;
  QByteArray dat;
  QVector<QColor> colors;
  QVector<double> last_change_t;
  QVector<std::array<uint32_t, 8>> bit_change_counts;
  QVector<int> last_delta;
  QVector<int> same_delta_counter;
};

struct CanEvent {
  uint64_t mono_time;
  uint8_t size;
  uint8_t dat[];
};

class AbstractStream : public QObject {
  Q_OBJECT

public:
  AbstractStream(QObject *parent, bool is_live_streaming);
  virtual ~AbstractStream() {};
  inline bool liveStreaming() const { return is_live_streaming; }
  inline double lastEventSecond() const { return last_event_ts / 1e9 - routeStartTime(); }
  virtual void seekTo(double ts) {}
  virtual QString routeName() const = 0;
  virtual QString carFingerprint() const { return ""; }
  virtual double totalSeconds() const { return 0; }
  virtual double routeStartTime() const { return 0; }
  virtual double currentSec() const = 0;
  virtual QDateTime currentDateTime() const { return {}; }
  virtual const CanData &lastMessage(const MessageId &id);
  virtual VisionStreamType visionStreamType() const { return VISION_STREAM_ROAD; }
  virtual const Route *route() const { return nullptr; }
  virtual void setSpeed(float speed) {}
  virtual double getSpeed() { return 1; }
  virtual bool isPaused() const { return false; }
  virtual void pause(bool pause) {}
  virtual const std::vector<Event*> *rawEvents() const { return nullptr; }
  const std::unordered_map<MessageId, std::deque<CanEvent *>> &events() const { return events_; }
  virtual const std::vector<std::tuple<int, int, TimelineType>> getTimeline() { return {}; }
  void mergeEvents(std::vector<Event *>::const_iterator first, std::vector<Event *>::const_iterator last, bool append);

signals:
  void paused();
  void resume();
  void seekedTo(double sec);
  void streamStarted();
  void eventsMerged();
  void updated();
  void msgsReceived(const QHash<MessageId, CanData> *);
  void received(QHash<MessageId, CanData> *);
  void sourcesUpdated(const SourceSet &s);

public:
  QHash<MessageId, CanData> last_msgs;
  SourceSet sources;

protected:
  virtual void process(QHash<MessageId, CanData> *);
  bool updateEvent(const Event *event);
  void updateLastMsgsTo(double sec);
  void parseEvents(std::unordered_map<MessageId, std::deque<CanEvent *>> &msgs, std::vector<Event *>::const_iterator first, std::vector<Event *>::const_iterator last);

  bool is_live_streaming = false;
  std::atomic<bool> processing = false;
  std::unique_ptr<QHash<MessageId, CanData>> new_msgs;
  QHash<MessageId, CanData> all_msgs;
  std::unordered_map<MessageId, std::deque<CanEvent *>> events_;
  uint64_t last_event_ts = 0;
  std::deque<std::unique_ptr<char[]>> memory_blocks;
};

class AbstractOpenStreamWidget : public QWidget {
  Q_OBJECT
public:
  AbstractOpenStreamWidget(AbstractStream **stream, QWidget *parent = nullptr) : stream(stream), QWidget(parent) {}
  virtual bool open() = 0;
  virtual QString title() = 0;

protected:
  AbstractStream **stream = nullptr;
};

// A global pointer referring to the unique AbstractStream object
extern AbstractStream *can;
