#pragma once

#include <vector>
#include <unordered_map>

#include <QElapsedTimer>
#include <QMultiMap>
#include <QNetworkAccessManager>
#include <QString>
#include <QThread>

#include <capnp/serialize.h>

#include "cereal/gen/cpp/log.capnp.h"

class FileReader : public QObject {
  Q_OBJECT

public:
  FileReader(const QString &fn, QObject *parent);
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

enum CameraType {
  RoadCam = 0,
  DriverCam,
  WideRoadCam
};
const CameraType ALL_CAMERAS[] = {RoadCam, DriverCam, WideRoadCam};
const int MAX_CAMERAS = std::size(ALL_CAMERAS);

struct EncodeIdx {
  int segmentNum;
  uint32_t segmentId;
};

class Event {
 public:
  Event(const kj::ArrayPtr<const capnp::word> &amsg) : reader(amsg) {
    words = kj::ArrayPtr<const capnp::word>(amsg.begin(), reader.getEnd());
    event = reader.getRoot<cereal::Event>();
    mono_time = event.getLogMonoTime();
  }
  inline kj::ArrayPtr<const capnp::byte> bytes() const { return words.asBytes(); }

  uint64_t mono_time;
  kj::ArrayPtr<const capnp::word> words;
  capnp::FlatArrayMessageReader reader;
  cereal::Event::Reader event;
};

typedef std::vector<Event*> Events;
typedef std::unordered_map<uint32_t, EncodeIdx> EncodeIdxMap;

class LogReader : public QObject {
  Q_OBJECT

public:
  LogReader(const QString &file);
  ~LogReader();
  inline bool valid() const { return valid_; }
  inline const Events &events() const { return events_; }
  const EncodeIdx *getFrameEncodeIdx(CameraType type, uint32_t frame_id) const {
    auto it = encoderIdx_[type].find(frame_id);
    return it != encoderIdx_[type].end() ? &it->second : nullptr;
  }

signals:
  void finished(bool success);

private:
  void start();
  void fileReady(const QByteArray &dat);
  void parseEvents(kj::ArrayPtr<const capnp::word> words);

  FileReader *file_reader_ = nullptr;
  std::vector<uint8_t> raw_;
  Events events_;
  EncodeIdxMap encoderIdx_[MAX_CAMERAS] = {};

  std::atomic<bool> exit_ = false;
  std::atomic<bool> valid_ = false;
  QThread *thread_ = nullptr;
};
