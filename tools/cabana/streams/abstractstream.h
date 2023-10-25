#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <QColor>
#include <QDateTime>
#include <QHash>

#include "common/timing.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/util.h"
#include "tools/replay/replay.h"

struct CanData {
  void compute(const MessageId &msg_id, const char *dat, const int size, double current_sec,
               double playback_speed, const std::vector<uint8_t> *mask = nullptr, double in_freq = 0);

  double ts = 0.;
  uint32_t count = 0;
  double freq = 0;
  QByteArray dat;
  QVector<QColor> colors;
  std::vector<double> last_change_t;
  std::vector<std::array<uint32_t, 8>> bit_change_counts;
  std::vector<int> last_delta;
  std::vector<int> same_delta_counter;
  double last_freq_update_ts = 0;
};

struct CanEvent {
  uint8_t src;
  uint32_t address;
  uint64_t mono_time;
  uint8_t size;
  uint8_t dat[];
};

struct CompareCanEvent {
  constexpr bool operator()(const CanEvent *const e, uint64_t ts) const { return e->mono_time < ts; }
  constexpr bool operator()(uint64_t ts, const CanEvent *const e) const { return ts < e->mono_time; }
};

struct BusConfig {
  int can_speed_kbps = 500;
  int data_speed_kbps = 2000;
  bool can_fd = false;
};

typedef std::unordered_map<MessageId, std::vector<const CanEvent *>> MessageEventsMap;

class AbstractStream : public QObject {
  Q_OBJECT

public:
  AbstractStream(QObject *parent);
  virtual ~AbstractStream() {}
  virtual void start() = 0;
  inline bool liveStreaming() const { return route() == nullptr; }
  virtual void seekTo(double ts) {}
  virtual QString routeName() const = 0;
  virtual QString carFingerprint() const { return ""; }
  virtual QDateTime beginDateTime() const { return {}; }
  virtual double routeStartTime() const { return 0; }
  virtual double currentSec() const = 0;
  virtual double totalSeconds() const { return lastEventMonoTime() / 1e9 - routeStartTime(); }
  const CanData &lastMessage(const MessageId &id);
  virtual VisionStreamType visionStreamType() const { return VISION_STREAM_ROAD; }
  virtual const Route *route() const { return nullptr; }
  virtual void setSpeed(float speed) {}
  virtual double getSpeed() { return 1; }
  virtual bool isPaused() const { return false; }
  virtual void pause(bool pause) {}
  const MessageEventsMap &eventsMap() const { return events_; }
  const std::vector<const CanEvent *> &allEvents() const { return all_events_; }
  const std::vector<const CanEvent *> &events(const MessageId &id) const;
  virtual const std::vector<std::tuple<double, double, TimelineType>> getTimeline() { return {}; }

signals:
  void paused();
  void resume();
  void seekedTo(double sec);
  void streamStarted();
  void eventsMerged(const MessageEventsMap &events_map);
  void updated();
  void msgsReceived(const QHash<MessageId, CanData> *new_msgs, bool has_new_ids);
  void sourcesUpdated(const SourceSet &s);

public:
  QHash<MessageId, CanData> last_msgs;
  SourceSet sources;

protected:
  void mergeEvents(std::vector<Event *>::const_iterator first, std::vector<Event *>::const_iterator last);
  bool postEvents();
  uint64_t lastEventMonoTime() const { return lastest_event_ts; }
  void updateEvent(const MessageId &id, double sec, const uint8_t *data, uint8_t size);
  void updateMessages(QHash<MessageId, CanData> *);
  void updateMasks();
  void updateLastMsgsTo(double sec);

  uint64_t lastest_event_ts = 0;
  std::atomic<bool> processing = false;
  std::unique_ptr<QHash<MessageId, CanData>> new_msgs;
  QHash<MessageId, CanData> all_msgs;
  MessageEventsMap events_;
  std::vector<const CanEvent *> all_events_;
  std::unique_ptr<MonotonicBuffer> event_buffer;
  std::mutex mutex;
  std::unordered_map<MessageId, std::vector<uint8_t>> masks;
};

class AbstractOpenStreamWidget : public QWidget {
public:
  AbstractOpenStreamWidget(AbstractStream **stream, QWidget *parent = nullptr) : stream(stream), QWidget(parent) {}
  virtual bool open() = 0;
  virtual QString title() = 0;

protected:
  AbstractStream **stream = nullptr;
};

class DummyStream : public AbstractStream {
  Q_OBJECT
public:
  DummyStream(QObject *parent) : AbstractStream(parent) {}
  QString routeName() const override { return tr("No Stream"); }
  void start() override { emit streamStarted(); }
  double currentSec() const override { return 0; }
};

class StreamNotifier : public QObject {
  Q_OBJECT
public:
  StreamNotifier(QObject *parent = nullptr) : QObject(parent) {}
  static StreamNotifier* instance();
signals:
  void streamStarted();
  void changingStream();
};

// A global pointer referring to the unique AbstractStream object
extern AbstractStream *can;
