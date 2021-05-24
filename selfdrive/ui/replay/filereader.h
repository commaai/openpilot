#pragma once

#include <vector>

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
  virtual void done() {};

public slots:
  void process();

protected:
  QNetworkReply *reply;

private:
  QNetworkAccessManager *qnam;
  QElapsedTimer timer;
  QString file;
};

typedef QMultiMap<uint64_t, capnp::FlatArrayMessageReader *> Events;
typedef QMap<int, QPair<int, int> > EncodeIdxMap;

class LogReader : public FileReader {
  Q_OBJECT

public:
  LogReader(const QString &file);
  ~LogReader();
  bool ready() const { return ready_; }
  const Events &events() const { return events_; }
  const EncodeIdxMap &roadCamEncodeIdx() const { return roadCamEncodeIdx_; }
  const EncodeIdxMap &driverCamEncodeIdx() const { return driverCamEncodeIdx_; }

signals:
  void done();

protected:
  void readyRead();
  void parseEvents(kj::ArrayPtr<const capnp::word> amsg);

  std::vector<uint8_t> raw_;
  Events events_;
  std::atomic<bool> ready_ = false;
  EncodeIdxMap roadCamEncodeIdx_;
  EncodeIdxMap driverCamEncodeIdx_;
};
