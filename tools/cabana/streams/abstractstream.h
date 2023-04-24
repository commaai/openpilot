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
  uint8_t src;
  uint32_t address;
  uint64_t mono_time;
  uint8_t size;
  uint8_t dat[];
};

class AbstractStream : public QObject {
  Q_OBJECT

public:
  AbstractStream(QObject *parent);
  virtual ~AbstractStream() {};
  inline bool liveStreaming() const { return route() == nullptr; }
  inline double lastEventSecond() const { return lastEventMonoTime() / 1e9 - routeStartTime(); }
  virtual void seekTo(double ts) {}
  virtual QString routeName() const = 0;
  virtual QString carFingerprint() const { return ""; }
  virtual double routeStartTime() const { return 0; }
  virtual double currentSec() const = 0;
  double totalSeconds() const { return total_sec; }
  const CanData &lastMessage(const MessageId &id);
  virtual VisionStreamType visionStreamType() const { return VISION_STREAM_ROAD; }
  virtual const Route *route() const { return nullptr; }
  virtual void setSpeed(float speed) {}
  virtual double getSpeed() { return 1; }
  virtual bool isPaused() const { return false; }
  virtual void pause(bool pause) {}
  const std::deque<const CanEvent *> &allEvents() const { return all_events_; }
  const std::deque<const CanEvent *> &events(const MessageId &id) const { return events_.at(id); }
  virtual const std::vector<std::tuple<int, int, TimelineType>> getTimeline() { return {}; }

signals:
  void paused();
  void resume();
  void seekedTo(double sec);
  void streamStarted();
  void eventsMerged();
  void updated();
  void msgsReceived(const QHash<MessageId, CanData> *);
  void sourcesUpdated(const SourceSet &s);

public:
  QHash<MessageId, CanData> last_msgs;
  SourceSet sources;

protected:
  void mergeEvents(std::vector<Event *>::const_iterator first, std::vector<Event *>::const_iterator last, bool append);
  bool postEvents();
  uint64_t lastEventMonoTime() const { return all_events_.empty() ? 0 : all_events_.back()->mono_time; }
  void updateEvent(const MessageId &id, double sec, const uint8_t *data, uint8_t size);
  void updateMessages(QHash<MessageId, CanData> *);
  void parseEvents(std::unordered_map<MessageId, std::deque<const CanEvent *>> &msgs, std::vector<Event *>::const_iterator first, std::vector<Event *>::const_iterator last);
  void updateLastMsgsTo(double sec);

  double total_sec = 0;
  std::atomic<bool> processing = false;
  std::unique_ptr<QHash<MessageId, CanData>> new_msgs;
  QHash<MessageId, CanData> all_msgs;
  std::unordered_map<MessageId, std::deque<const CanEvent *>> events_;
  std::deque<const CanEvent *> all_events_;
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
