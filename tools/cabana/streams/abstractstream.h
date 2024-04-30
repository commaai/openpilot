#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include <QColor>
#include <QDateTime>

#include "cereal/messaging/messaging.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/utils/util.h"
#include "tools/replay/util.h"

struct CanData {
  void compute(const MessageId &msg_id, const uint8_t *dat, const int size, double current_sec,
               double playback_speed, const std::vector<uint8_t> &mask, double in_freq = 0);

  double ts = 0.;
  uint32_t count = 0;
  double freq = 0;
  std::vector<uint8_t> dat;
  std::vector<QColor> colors;

  struct ByteLastChange {
    double ts;
    int delta;
    int same_delta_counter;
    bool suppressed;
    std::array<uint32_t, 8> bit_change_counts;
  };
  std::vector<ByteLastChange> last_changes;
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
  virtual bool liveStreaming() const { return true; }
  virtual void seekTo(double ts) {}
  virtual QString routeName() const = 0;
  virtual QString carFingerprint() const { return ""; }
  virtual QDateTime beginDateTime() const { return {}; }
  virtual double routeStartTime() const { return 0; }
  inline double currentSec() const { return current_sec_; }
  virtual double totalSeconds() const { return lastEventMonoTime() / 1e9 - routeStartTime(); }
  virtual void setSpeed(float speed) {}
  virtual double getSpeed() { return 1; }
  virtual bool isPaused() const { return false; }
  virtual void pause(bool pause) {}

  inline const std::unordered_map<MessageId, CanData> &lastMessages() const { return last_msgs; }
  inline const MessageEventsMap &eventsMap() const { return events_; }
  inline const std::vector<const CanEvent *> &allEvents() const { return all_events_; }
  const CanData &lastMessage(const MessageId &id);
  const std::vector<const CanEvent *> &events(const MessageId &id) const;

  size_t suppressHighlighted();
  void clearSuppressed();
  void suppressDefinedSignals(bool suppress);

signals:
  void paused();
  void resume();
  void seekedTo(double sec);
  void streamStarted();
  void eventsMerged(const MessageEventsMap &events_map);
  void msgsReceived(const std::set<MessageId> *new_msgs, bool has_new_ids);
  void sourcesUpdated(const SourceSet &s);
  void privateUpdateLastMsgsSignal();

public:
  SourceSet sources;

protected:
  void mergeEvents(const std::vector<const CanEvent *> &events);
  const CanEvent *newEvent(uint64_t mono_time, const cereal::CanData::Reader &c);
  void updateEvent(const MessageId &id, double sec, const uint8_t *data, uint8_t size);
  uint64_t lastEventMonoTime() const { return lastest_event_ts; }

  std::vector<const CanEvent *> all_events_;
  uint64_t lastest_event_ts = 0;

private:
  void updateLastMessages();
  void updateLastMsgsTo(double sec);
  void updateMasks();

  double current_sec_ = 0;
  MessageEventsMap events_;
  std::unordered_map<MessageId, CanData> last_msgs;
  std::unique_ptr<MonotonicBuffer> event_buffer_;

  // Members accessed in multiple threads. (mutex protected)
  std::mutex mutex_;
  std::set<MessageId> new_msgs_;
  std::unordered_map<MessageId, CanData> messages_;
  std::unordered_map<MessageId, std::vector<uint8_t>> masks_;
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
