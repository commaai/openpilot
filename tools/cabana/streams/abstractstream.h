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
  void compute(const char *dat, const int size, double current_sec, double playback_speed, const std::vector<uint8_t> *mask, uint32_t in_freq = 0);

  double ts = 0.;
  uint32_t count = 0;
  double freq = 0;
  QByteArray dat;
  QVector<QColor> colors;
  std::vector<double> last_change_t;
  std::vector<std::array<uint32_t, 8>> bit_change_counts;
  std::vector<int> last_delta;
  std::vector<int> same_delta_counter;
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
  virtual ~AbstractStream() {}
  virtual void start() = 0;
  inline bool liveStreaming() const { return route() == nullptr; }
  virtual void seekTo(double ts) {}
  virtual QString routeName() const = 0;
  virtual QString carFingerprint() const { return ""; }
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
  const std::vector<const CanEvent *> &allEvents() const { return all_events_; }
  const std::vector<const CanEvent *> &events(const MessageId &id) const;
  virtual const std::vector<std::tuple<double, double, TimelineType>> getTimeline() { return {}; }

signals:
  void paused();
  void resume();
  void seekedTo(double sec);
  void streamStarted();
  void eventsMerged();
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
  std::unordered_map<MessageId, std::vector<const CanEvent *>> events_;
  std::vector<const CanEvent *> all_events_;
  std::deque<std::unique_ptr<char[]>> memory_blocks;
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
