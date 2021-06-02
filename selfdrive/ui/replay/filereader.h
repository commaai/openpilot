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
  Event(const kj::ArrayPtr<const capnp::word> &amsg, std::shared_ptr<std::vector<uint8_t>> raw) : reader(amsg), raw(raw) {
    words = kj::ArrayPtr<const capnp::word>(amsg.begin(), reader.getEnd());
    event = reader.getRoot<cereal::Event>();
  }
  // ~Event() {
  //   if (raw.use_count() == 1) {
  //     qDebug() << "delete raw" << i++;
  //   }
  // }
  inline kj::ArrayPtr<const capnp::byte> bytes() const { return words.asBytes(); }

  kj::ArrayPtr<const capnp::word> words;
  capnp::FlatArrayMessageReader reader;
  cereal::Event::Reader event;
private:
  std::shared_ptr<std::vector<uint8_t>> raw;
  inline static int i = 0;
};

typedef QMultiMap<uint64_t, Event *> Events;
typedef std::unordered_map<uint32_t, EncodeIdx> EncodeIdxMap;

class LogReader : public QObject {
  Q_OBJECT

public:
  LogReader(const QString &file);
  ~LogReader();
  inline bool valid() const { return valid_; }

  Events events;
  EncodeIdxMap encoderIdx[MAX_CAMERAS] = {};

signals:
  void finished(bool success);

private:
  void start();
  void parseEvents(const QByteArray &dat);

  FileReader *file_reader_ = nullptr;

  std::atomic<bool> exit_ = false;
  std::atomic<bool> valid_ = false;
  QThread *thread_ = nullptr;
};
