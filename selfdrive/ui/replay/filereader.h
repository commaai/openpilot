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
  void startRequest(const QUrl &url);
  virtual void readyRead();
  void httpFinished();
  virtual void done() {}

public slots:
  void process();

protected:
  QNetworkReply *reply;

private:
  QNetworkAccessManager *qnam;
  QElapsedTimer timer;
  QString file;
};

typedef QMultiMap<uint64_t, capnp::FlatArrayMessageReader*> Events;

class LogReader : public FileReader {
  Q_OBJECT

public:
  LogReader(const QString &file, Events *, QReadWriteLock* events_lock_, QMap<int, QPair<int, int> > *eidx_);
  void readyRead();
  void done() { is_done = true; };
  bool is_done = false;

private:
  void mergeEvents(kj::ArrayPtr<const capnp::word> amsg);

  // backing store
  std::vector<uint8_t> raw;

  Events *events;
  QReadWriteLock* events_lock;
  QMap<int, QPair<int, int> > *eidx;
};
