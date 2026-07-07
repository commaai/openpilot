#pragma once

// Threading contract for AbstractStream and its subclasses:
//
// Producer threads (replay's event filter thread, live-stream reader
// threads) only ever stage data under mutex_ (updateEvent() -> messages_ /
// new_msgs_) or enqueue a deferred action via enqueue(). They never touch
// last_msgs, events_, all_events_, or fire an Event<> directly.
//
// The UI thread calls AbstractStream::update() once per frame. update() (a)
// drains the deferred-action queue every call, and (b) refreshes last_msgs
// and fires data-changed events at a settings.fps-throttled cadence. This
// replaces the old producer-side privateUpdateLastMsgsSignal throttle plus
// the Qt::QueuedConnection hop to the UI thread.
//
// All Event<> callbacks fire on the UI thread only -- either from update()
// or from direct UI-thread API calls (setTimeRange, pause, seekTo, ...).

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "openpilot/cereal/messaging/messaging.h"
#include "tools/cabana/dbc/dbc.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/utils/event.h"
#include "tools/replay/util.h"

struct CanData {
  void compute(const MessageId &msg_id, const uint8_t *dat, const int size, double current_sec,
               double playback_speed, const std::vector<uint8_t> &mask, double in_freq = 0);

  double ts = 0.;
  uint32_t count = 0;
  double freq = 0;
  std::vector<uint8_t> dat;
  std::vector<ColorRGBA> colors;

  struct ByteLastChange {
    double ts = 0;
    int delta = 0;
    int same_delta_counter = 0;
    bool suppressed = false;
  };
  std::vector<ByteLastChange> last_changes;
  std::vector<std::array<uint32_t, 8>> bit_flip_counts;
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
using CanEventIter = std::vector<const CanEvent *>::const_iterator;

class AbstractStream {
public:
  AbstractStream();
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
  virtual float getSpeed() const { return 1; }
  virtual bool isPaused() const { return false; }
  virtual void pause(bool pause) {}
  virtual void update();
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

  cabana::Event<> paused;
  cabana::Event<> resume;
  cabana::Event<double> seeking;
  cabana::Event<double> seekedTo;
  cabana::Event<const std::optional<std::pair<double, double>> &> timeRangeChanged;
  cabana::Event<const MessageEventsMap &> eventsMerged;
  cabana::Event<const std::set<MessageId> *, bool> msgsReceived;
  cabana::Event<const SourceSet &> sourcesUpdated;

  SourceSet sources;

protected:
  void mergeEvents(const std::vector<const CanEvent *> &events);
  const CanEvent *newEvent(uint64_t mono_time, const cereal::CanData::Reader &c);
  void updateEvent(const MessageId &id, double sec, const uint8_t *data, uint8_t size);
  void waitForSeekFinshed();
  void enqueue(std::function<void()> action);
  // True if called from the thread that constructed this stream (the UI
  // thread). Some Replay callbacks fire synchronously on the calling thread
  // when their result is already available (e.g. seeking into an
  // already-loaded segment) and asynchronously from a replay-owned thread
  // otherwise; subclasses use this to run the UI-thread case inline instead
  // of enqueueing (which would deadlock: nothing would ever drain the queue).
  bool onUiThread() const { return std::this_thread::get_id() == ui_thread_id_; }
  void updateLastMsgsTo(double sec);
  std::vector<const CanEvent *> all_events_;
  double current_sec_ = 0;
  std::optional<std::pair<double, double>> time_range_;

private:
  void updateLastMessages();
  void updateMasks();

  MessageEventsMap events_;
  std::unordered_map<MessageId, CanData> last_msgs;
  std::unique_ptr<MonotonicBuffer> event_buffer_;
  double last_update_ts_ = 0;
  std::thread::id ui_thread_id_;

  // Members accessed in multiple threads. (mutex protected)
  std::mutex mutex_;
  std::condition_variable seek_finished_cv_;
  bool seek_finished_ = false;
  std::set<MessageId> new_msgs_;
  std::unordered_map<MessageId, CanData> messages_;
  std::unordered_map<MessageId, std::vector<uint8_t>> masks_;

  std::mutex actions_mutex_;
  std::deque<std::function<void()>> actions_;
};

class DummyStream : public AbstractStream {
public:
  DummyStream() : AbstractStream() {}
  std::string routeName() const override { return "No Stream"; }
  void start() override {}
};

// A global pointer referring to the unique AbstractStream object
extern AbstractStream *can;
