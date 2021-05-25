#pragma once

#include <optional>
#include <vector>
#include <unordered_map>

#include <QElapsedTimer>
#include <QMultiMap>
#include <QNetworkAccessManager>
#include <QString>

#include <capnp/serialize.h>

class FileReader : public QObject {
  Q_OBJECT

public:
  FileReader(const QString& file_);
  void startRequest(const QUrl &url);
  ~FileReader();
  virtual void readyRead();
  void httpFinished();

public slots:
  void process();

protected:
  QNetworkReply *reply;

private:
  QNetworkAccessManager *qnam;
  QElapsedTimer timer;
  QString file;
};

struct EncodeIdx {
  int segmentNum;
  uint32_t segmentId;
};

typedef QMultiMap<uint64_t, capnp::FlatArrayMessageReader *> Events;
typedef std::unordered_map<int, EncodeIdx> EncodeIdxMap;

class LogReader : public FileReader {
  Q_OBJECT

public:
  LogReader(const QString &file);
  ~LogReader();
  bool ready() const { return ready_; }
  const Events &events() const { return events_; }
  std::optional<EncodeIdx> getFrameEncodeIdx(const std::string &type, uint32_t frame_id) const;

signals:
  void done();

protected:
  void readyRead();
  void parseEvents(kj::ArrayPtr<const capnp::word> amsg);

  std::vector<uint8_t> raw_;
  Events events_;
  std::atomic<bool> ready_ = false;
  std::map<std::string, EncodeIdxMap> encoderIdx_;
};
