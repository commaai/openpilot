#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
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
    double ts = 0;
    int delta = 0;
    int same_delta_counter = 0;
    bool suppressed = false;
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
  virtual uint64_t beginMonoTime() const { return 0; }
  virtual double minSeconds() const { return 0; }
  virtual double maxSeconds() const { return 0; }
  virtual void setSpeed(float speed) {}
  virtual double getSpeed() { return 1; }
  virtual bool isPaused() const { return false; }
  virtual void pause(bool pause) {}
  void setTimeRange(const std::optional<std::pair<double, double>> &range);
  const std::optional<std::pair<double, double>> &timeRange() const { return time_range_; }

  inline double currentSec() const { return current_sec_; }
  inline uint64_t toMonoTime(double sec) const { return beginMonoTime() + std::max(sec, 0.0) * 1e9; }
  inline double toSeconds(uint64_t mono_time) const { return std::max(0.0, (mono_time - beginMonoTime()) / 1e9); }

  inline const std::unordered_map<MessageId, CanData> &lastMessages() const { return last_msgs; }
  inline const MessageEventsMap &eventsMap() const { return events_; }
  inline const std::vector<const CanEvent *> &allEvents() const { return all_events_; }
  const CanData &lastMessage(const MessageId &id) const;
  const std::vector<const CanEvent *> &events(const MessageId &id) const;

  size_t suppressHighlighted();
  void clearSuppressed();
  void suppressDefinedSignals(bool suppress);

signals:
  void paused();
  void resume();
  void seeking(double sec);
  void seekedTo(double sec);
  void timeRangeChanged(const std::optional<std::pair<double, double>> &range);
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

  std::vector<const CanEvent *> all_events_;
  double current_sec_ = 0;
  std::optional<std::pair<double, double>> time_range_;

private:
  void updateLastMessages();
  void updateLastMsgsTo(double sec);
  void updateMasks();

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
  Q_OBJECT
public:
  AbstractOpenStreamWidget(QWidget *parent = nullptr) : QWidget(parent) {}
  virtual AbstractStream *open() = 0;

signals:
  void enableOpenButton(bool);
};

class DummyStream : public AbstractStream {
  Q_OBJECT
public:
  DummyStream(QObject *parent) : AbstractStream(parent) {}
  QString routeName() const override { return tr("No Stream"); }
  void start() override {}
};

// A global pointer referring to the unique AbstractStream object
extern AbstractStream *can;
