#include "selfdrive/ui/replay/filereader.h"

#include <QtNetwork>

FileReader::FileReader(const QString& file_) : file(file_) {
  qnam = new QNetworkAccessManager(this);
}

void FileReader::process() {
  timer.start();
  QString str = file.simplified();
  str.replace(" ", "");
  startRequest(QUrl(str));
}

void FileReader::startRequest(const QUrl &url) {
  reply = qnam->get(QNetworkRequest(url));
  connect(reply, &QNetworkReply::finished, this, &FileReader::httpFinished);
  connect(reply, &QIODevice::readyRead, this, &FileReader::readyRead);
  qDebug() << "requesting" << url;
}

void FileReader::httpFinished() {
  if (reply->error()) {
    qWarning() << reply->errorString();
  }

  const QVariant redirectionTarget = reply->attribute(QNetworkRequest::RedirectionTargetAttribute);
  if (!redirectionTarget.isNull()) {
    const QUrl redirectedUrl = redirectionTarget.toUrl();
    //qDebug() << "redirected to" << redirectedUrl;
    startRequest(redirectedUrl);
  } else {
    qDebug() << "done in" << timer.elapsed() << "ms";
    done();
  }
  reply->deleteLater();
  reply = nullptr;
}

void FileReader::readyRead() {
  QByteArray dat = reply->readAll();
  printf("got http ready read: %d\n", dat.size());
}

LogReader::LogReader(const QString& file, Events *events_, QReadWriteLock* events_lock_, QMap<int, QPair<int, int> > *eidx_) :
    FileReader(file), events(events_), events_lock(events_lock_), eidx(eidx_) {
  int ret = BZ2_bzDecompressInit(&bStream, 0, 0);
  assert(ret == BZ_OK);

  // start with 64MB buffer
  raw.resize(1024*1024*64);

  // auto increment?
  bStream.next_out = raw.data();
  bStream.avail_out = raw.size();
}

LogReader::~LogReader() {
  BZ2_bzDecompressEnd(&bStream);
}

void LogReader::mergeEvents(kj::ArrayPtr<const capnp::word> amsg) {
  Events events_local;
  QMap<int, QPair<int, int> > eidx_local;
  size_t offset = 0;
  while (offset < amsg.size()) {
    try {
      std::unique_ptr<capnp::FlatArrayMessageReader> reader =
          std::make_unique<capnp::FlatArrayMessageReader>(amsg.slice(offset, amsg.size()));

      cereal::Event::Reader event = reader->getRoot<cereal::Event>();
      offset = reader->getEnd() - amsg.begin();

      // hack
      // TODO: rewrite with callback
      if (event.which() == cereal::Event::ROAD_ENCODE_IDX) {
        auto ee = event.getRoadEncodeIdx();
        eidx_local.insert(ee.getFrameId(), qMakePair(ee.getSegmentNum(), ee.getSegmentId()));
      }

      events_local.insert(event.getLogMonoTime(), reader.release());
    } catch (const kj::Exception& e) {
      // partial messages trigger this
      // qDebug() << e.getDescription().cStr();
      break;
    }
  }

  // merge in events
  // TODO: add lock
  events_lock->lockForWrite();
  *events += events_local;
  eidx->unite(eidx_local);
  events_lock->unlock();
}

void LogReader::readyRead() {
  QByteArray dat = reply->readAll();

  bStream.next_in = dat.data();
  bStream.avail_in = dat.size();

  while (bStream.avail_in > 0) {
    int ret = BZ2_bzDecompress(&bStream);
    if (ret != BZ_OK && ret != BZ_STREAM_END) {
      qWarning() << "bz2 decompress failed";
      break;
    }
  }
  size_t size = (raw.size() - bStream.avail_out) / sizeof(capnp::word);
  mergeEvents({(const capnp::word*)raw.data(), size});
}
