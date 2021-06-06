#pragma once

#include <capnp/serialize.h>

#include <QElapsedTimer>
#include <QMultiMap>
#include <QNetworkAccessManager>
#include <QString>
#include <QThread>
#include <unordered_map>
#include <vector>

#include "cereal/gen/cpp/log.capnp.h"

#include "selfdrive/ui/replay/camera.h"

class FileReader : public QObject {
  Q_OBJECT

public:
  FileReader(const QString &fn, QObject *parent = nullptr);
  void read();
  void abort();

signals:
  void finished(const QByteArray &dat);
  void failed(const QString &err);

private:
  void startHttpRequest();
  QNetworkReply *reply_ = nullptr;
  QUrl url_;
};

struct EncodeIdx {
  int segmentNum;
  uint32_t segmentId;
};

class Event {
public:
  Event(const kj::ArrayPtr<const capnp::word> &amsg) : reader(amsg) {
    words = kj::ArrayPtr<const capnp::word>(amsg.begin(), reader.getEnd());
    event = reader.getRoot<cereal::Event>();
    which = event.which();
    mono_time = event.getLogMonoTime();
  }
  friend inline bool operator<(const Event& l, const Event& r) {
    return l.mono_time < r.mono_time || (l.mono_time == r.mono_time && l.which < r.which);
  }

  inline kj::ArrayPtr<const capnp::byte> bytes() const { return words.asBytes(); }

  uint64_t mono_time;
  cereal::Event::Which which;
  cereal::Event::Reader event;
  capnp::FlatArrayMessageReader reader;
  kj::ArrayPtr<const capnp::word> words;
};

typedef std::unordered_map<uint32_t, EncodeIdx> EncodeIdxMap;

class LogReader : public QObject {
  Q_OBJECT

public:
  LogReader(const QString &file, QObject *parent = nullptr);
  ~LogReader();
  inline bool valid() const { return valid_; }

  // Events events;
  std::vector<Event *> events;
  uint64_t route_start_ts = 0;
  EncodeIdxMap encoderIdx[MAX_CAMERAS] = {};

signals:
  void finished(bool success);

private:
  void parseEvents(const QByteArray &dat);

  std::atomic<bool> exit_ = false;
  std::atomic<bool> valid_ = false;
  std::vector<uint8_t> raw_;

  FileReader *file_reader_ = nullptr;
  QThread thread_;
};
