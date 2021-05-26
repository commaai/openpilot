#pragma once

#include <vector>
#include <unordered_map>

#include <QElapsedTimer>
#include <QMultiMap>
#include <QNetworkAccessManager>
#include <QString>
#include <QThread>

#include <capnp/serialize.h>

class FileReader : public QObject {
  Q_OBJECT

public:
  void startRequest(const QUrl &url);
  
signals:
  void ready(const QByteArray &dat);

protected:
  void readyRead();
  void httpFinished();
  QNetworkReply *reply;

private:
  QNetworkAccessManager *qnam;
  QElapsedTimer timer;
};

enum FrameType {
  RoadCamFrame = 0,
  DriverCamFrame,
  WideRoadCamFrame
};
const int MAX_FRAME_TYPE = WideRoadCamFrame + 1;

struct EncodeIdx {
  int segmentNum;
  uint32_t segmentId;
};

struct EventMsg {
  public:
  EventMsg(const kj::ArrayPtr<const capnp::word> &amsg) : msg(amsg) {
    words = kj::ArrayPtr<const capnp::word>(amsg.begin(), msg.getEnd());
  }
  kj::ArrayPtr<const capnp::word> words;
  capnp::FlatArrayMessageReader msg;
};

typedef QMultiMap<uint64_t, EventMsg*> Events;
typedef std::unordered_map<int, EncodeIdx> EncodeIdxMap;

class LogReader : public QThread {
  Q_OBJECT

public:
  LogReader(const QString &file, QObject *parent);
  ~LogReader();
  void run() override;
  inline const Events &events() const { return events_; }
  const EncodeIdx *getFrameEncodeIdx(FrameType type, uint32_t frame_id) const {
    auto it = encoderIdx_[type].find(frame_id);
    return it != encoderIdx_[type].end() ? &it->second : nullptr;
  }

signals:
  void finished(bool success);

protected:
  void readyRead(const QByteArray &dat);
  void parseEvents(kj::ArrayPtr<const capnp::word> amsg);

  std::vector<uint8_t> raw_;
  Events events_;
  EncodeIdxMap encoderIdx_[MAX_FRAME_TYPE];
  QString file_;
  std::atomic<bool> exit_;
};
