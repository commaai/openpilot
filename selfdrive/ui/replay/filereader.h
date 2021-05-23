#pragma once

#include <vector>

#include <QElapsedTimer>
#include <QMultiMap>
#include <QNetworkAccessManager>
#include <QReadWriteLock>
#include <QString>

#include <capnp/serialize.h>

class FileReader : public QObject {
  Q_OBJECT

public:
  FileReader(const QString& file_);
  
public slots:
  void process();

protected:
  void startRequest(const QUrl &url);
  virtual void readyRead();
  void httpFinished();

  QNetworkReply *reply;
  QNetworkAccessManager *qnam;
  QElapsedTimer timer;
  QString file;
};

typedef QMultiMap<uint64_t, capnp::FlatArrayMessageReader*> Events;

class LogReader : public FileReader {
  Q_OBJECT

public:
  LogReader(const QString &file, Events *, QReadWriteLock* events_lock_, QMap<int, QPair<int, int> > *eidx_);
  
protected:
  void readyRead();
  void mergeEvents(kj::ArrayPtr<const capnp::word> amsg);

  // backing store
  std::vector<uint8_t> raw;

  Events *events;
  QReadWriteLock* events_lock;
  QMap<int, QPair<int, int> > *eidx;
};
