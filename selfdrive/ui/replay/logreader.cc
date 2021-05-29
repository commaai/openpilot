#include "selfdrive/ui/replay/logreader.h"

#include <QtNetwork>

FileReader::FileReader(const QString& file_) : file(file_) {
}

void FileReader::process() {
  timer.start();
  QString str = file.simplified();
  str.replace(" ", "");
  startRequest(QUrl(str));
}

void FileReader::startRequest(const QUrl &url) {
  qnam = new QNetworkAccessManager;
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
}

void FileReader::readyRead() {
  QByteArray dat = reply->readAll();
  printf("got http ready read: %d\n", dat.size());
}

FileReader::~FileReader() {

}

LogReader::LogReader(const QString& file, Events *events_, QReadWriteLock* events_lock_, QMap<int, QPair<int, int> > *eidx_) :
    FileReader(file), events(events_), events_lock(events_lock_), eidx(eidx_) {
  bStream.next_in = NULL;
  bStream.avail_in = 0;
  bStream.bzalloc = NULL;
  bStream.bzfree = NULL;
  bStream.opaque = NULL;

  int ret = BZ2_bzDecompressInit(&bStream, 0, 0);
  if (ret != BZ_OK) qWarning() << "bz2 init failed";

  // start with 64MB buffer
  raw.resize(1024*1024*64);

  // auto increment?
  bStream.next_out = raw.data();
  bStream.avail_out = raw.size();

  // parsed no events yet
  event_offset = 0;

  parser = new std::thread([&]() {
    while (1) {
      mergeEvents(cdled.get());
    }
  });
}

LogReader::~LogReader() {
  delete parser;
}

void LogReader::mergeEvents(int dled) {
  auto words = kj::arrayPtr((const capnp::word*)(raw.data() + event_offset), (dled-event_offset)/sizeof(capnp::word));
  Events events_local;
  QMap<int, QPair<int, int> > eidx_local;

  while (words.size() > 0) {
    try {
      std::unique_ptr<EventMsg> message = std::make_unique<EventMsg>(words);

      cereal::Event::Reader event = message->msg.getRoot<cereal::Event>();

      // hack
      // TODO: rewrite with callback
      if (event.which() == cereal::Event::ROAD_ENCODE_IDX) {
        auto ee = event.getRoadEncodeIdx();
        eidx_local.insert(ee.getFrameId(), qMakePair(ee.getSegmentNum(), ee.getSegmentId()));
      }

      // increment
      words = kj::arrayPtr(message->msg.getEnd(), words.end());
      event_offset = (char*)message->msg.getEnd() - raw.data();
      events_local.insert(event.getLogMonoTime(), message.release());
    } catch (const kj::Exception& e) {
      // partial messages trigger this
      //qDebug() << e.getDescription().cStr();
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

  int dled = raw.size() - bStream.avail_out;
  cdled.put(dled);
}

