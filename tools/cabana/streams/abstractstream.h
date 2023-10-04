#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <QColor>

#include "common/timing.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/util.h"
#include "tools/replay/replay.h"

struct CanData {
  void update(const MessageId &msg_id, const uint8_t *dat, const int size, uint64_t current_ts,
               double playback_speed, const std::vector<uint8_t> &mask);

  uint64_t mono_time = 0.;
  uint32_t count = 0;
  double freq = 0;
  std::vector<uint8_t> dat;
  std::vector<QColor> colors;

  struct ByteLastChange {
    uint64_t mono_time;
    int delta;
    int same_delta_counter;
    bool suppressed;
    std::array<uint32_t, 8> bit_change_counts;
  };

  std::vector<ByteLastChange> last_changes;
  uint64_t last_freq_update_ts = 0;
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

typedef std::unordered_map<MessageId, std::vector<const CanEvent *>> CanEventsMap;

class AbstractStream : public QObject {
  Q_OBJECT

public:
  AbstractStream(QObject *parent);
  virtual ~AbstractStream() {}
  virtual void start() = 0;
  inline bool liveStreaming() const { return route() == nullptr; }
  virtual void seekTo(double ts) {}
  virtual QString name() const = 0;
  virtual QString carFingerprint() const { return ""; }
  virtual uint64_t beginMonoTime() const { return 0; }
  double currentSec() const { return (currentMonoTime() - std::min(currentMonoTime(), beginMonoTime())) / 1e9; }
  virtual uint64_t currentMonoTime() const = 0;
  virtual double totalSeconds() const = 0;
  inline double toSeconds(uint64_t mono_time) const { return (mono_time - std::min(mono_time, beginMonoTime())) / 1e9; }
  inline uint64_t toMonoTime(double sec) const { return sec * 1e9 + beginMonoTime(); }
  const CanData &lastMessage(const MessageId &id);
  virtual VisionStreamType visionStreamType() const { return VISION_STREAM_ROAD; }
  virtual const Route *route() const { return nullptr; }
  virtual void setSpeed(float speed) {}
  virtual double getSpeed() { return 1; }
  virtual bool isPaused() const { return false; }
  virtual void pause(bool pause) {}
  inline const std::unordered_map<MessageId, CanData> &lastMessages() const { return last_msgs_; }
  inline const std::vector<const CanEvent *> &allEvents() const { return all_events_; }
  inline const CanEventsMap &eventsMap() const { return events_; }
  inline const SourceSet &sources() const { return sources_; }
  const std::vector<const CanEvent *> &events(const MessageId &id) const;
  virtual const std::vector<std::tuple<double, double, TimelineType>> getTimeline() { return {}; }
  size_t suppressHighlighted();
  void clearSuppressed();
  void suppressDefinedSignals(bool suppress);

signals:
  void paused();
  void resume();
  void seekedTo(double sec);
  void streamStarted();
  void eventsMerged(const CanEventsMap &can_events_map);
  void msgsReceived(const std::set<MessageId> *new_msgs, bool has_new_ids);
  void sourcesUpdated(const SourceSet &s);
  void lastMsgsChanged();
  void qLogLoaded(int segnum, std::shared_ptr<LogReader> qlog);

protected:
  void mergeEvents(const std::vector<const CanEvent *> &events);
  const CanEvent *newEvent(uint64_t mono_time, const cereal::CanData::Reader &c);
  void updateEvent(const MessageId &id, uint64_t ts, const uint8_t *data, uint8_t size);
  void updateLastMessages();
  void updateMasks();
  void updateLastMsgsTo(double sec);

  CanEventsMap events_;
  std::vector<const CanEvent *> all_events_;
  std::unique_ptr<MonotonicBuffer> event_buffer_;
  std::unordered_map<MessageId, CanData> last_msgs_;
  SourceSet sources_;

  // Members accessed in multiple threads. (mutex protected)
  std::recursive_mutex mutex_;
  std::set<MessageId> new_msgs_;
  std::unordered_map<MessageId, std::vector<uint8_t>> masks_;
  std::unordered_map<MessageId, CanData> msgs_;
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
  QString name() const override { return tr("No Stream"); }
  void start() override { emit streamStarted(); }
  uint64_t currentMonoTime() const override { return 0; }
  double totalSeconds() const override { return 0; }
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
