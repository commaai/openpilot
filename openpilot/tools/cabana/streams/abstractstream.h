#pragma once

#include <algorithm>
#include <array>
#include <condition_variable>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "openpilot/cereal/messaging/messaging.h"
#include "tools/cabana/core/can_data.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/utils/util.h"
#include "tools/replay/util.h"

class AbstractStream : public QObject {
  Q_OBJECT

public:
  AbstractStream(QObject *parent);
  virtual ~AbstractStream() {}
  virtual void start() = 0;
  virtual bool liveStreaming() const { return true; }
  virtual void seekTo(double ts) {}
  virtual std::string routeName() const = 0;
  virtual std::string carFingerprint() const { return ""; }
  virtual std::chrono::system_clock::time_point beginDateTime() const { return {}; }
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
  bool isMessageActive(const MessageId &id) const;
  inline const MessageEventsMap &eventsMap() const { return events_; }
  inline const std::vector<const CanEvent *> &allEvents() const { return all_events_; }
  const CanData &lastMessage(const MessageId &id) const;
  const std::vector<const CanEvent *> &events(const MessageId &id) const;
  std::pair<CanEventIter, CanEventIter> eventsInRange(const MessageId &id, std::optional<std::pair<double, double>> time_range) const;

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
  void waitForSeekFinshed();
  virtual void updateLastMessages();
  std::vector<const CanEvent *> all_events_;
  double current_sec_ = 0;
  std::optional<std::pair<double, double>> time_range_;

private:
  void updateLastMsgsTo(double sec);
  void updateMasks();

  MessageEventsMap events_;
  std::unordered_map<MessageId, CanData> last_msgs;
  std::unique_ptr<MonotonicBuffer> event_buffer_;

  // Members accessed in multiple threads. (mutex protected)
  std::mutex mutex_;
  std::condition_variable seek_finished_cv_;
  bool seek_finished_ = false;
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
  std::string routeName() const override { return "No Stream"; }
  void start() override {}
};

// A global pointer referring to the unique AbstractStream object
extern AbstractStream *can;
