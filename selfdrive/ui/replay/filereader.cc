#include "selfdrive/ui/replay/filereader.h"

#include <QtNetwork>

#include <bzlib.h>

static bool decompressBZ2(std::vector<uint8_t> &dest, const char srcData[], size_t srcSize,
                          size_t outputSizeIncrement = 0x100000U) {
  bz_stream strm = {};
  int ret = BZ2_bzDecompressInit(&strm, 0, 0);
  assert(ret == BZ_OK);

  strm.next_in = const_cast<char *>(srcData);
  strm.avail_in = srcSize;
  do {
    strm.next_out = (char *)&dest[strm.total_out_lo32];
    strm.avail_out = dest.size() - strm.total_out_lo32;
    ret = BZ2_bzDecompress(&strm);
    if (ret == BZ_OK && strm.avail_in > 0 && strm.avail_out == 0) {
      dest.resize(dest.size() + outputSizeIncrement);
    }
  } while (ret == BZ_OK);

  BZ2_bzDecompressEnd(&strm);
  dest.resize(strm.total_out_lo32);
  return ret == BZ_STREAM_END;
}

// class FileReader

FileReader::FileReader(const QString &fn, QObject *parent) : url_(fn), QObject(parent) {}

void FileReader::read() {
  if (url_.isLocalFile()) {
    QFile file(url_.toLocalFile());
    if (file.open(QIODevice::ReadOnly)) {
      emit finished(file.readAll());
    } else {
      emit failed(QString("Failed to read file %1").arg(url_.toString()));
    }
  } else {
    startHttpRequest();
  }
}

void FileReader::startHttpRequest() {
  QNetworkAccessManager *qnam = new QNetworkAccessManager(this);
  QNetworkRequest request(url_);
  request.setAttribute(QNetworkRequest::FollowRedirectsAttribute, true);
  reply_ = qnam->get(request);
  connect(reply_, &QNetworkReply::finished, [=]() {
    emit !reply_->error() ? finished(reply_->readAll()) : failed(reply_->errorString());
    reply_->deleteLater();
    reply_ = nullptr;
  });
}

void FileReader::abort() {
  if (reply_) reply_->abort();
}

// class LogReader

LogReader::LogReader(const QString& file, Events *events_, QReadWriteLock* events_lock_, QMap<int, QPair<int, int> > *eidx_) :
    events(events_), events_lock(events_lock_), eidx(eidx_) {
  file_reader_ = new FileReader(file, this);
  connect(file_reader_, &FileReader::finished, this, &LogReader::fileReady);
  connect(file_reader_, &FileReader::failed, [=](const QString &err) { qInfo() << err; });

  thread_ = new QThread(this);
  connect(thread_, &QThread::started, this, &LogReader::start);
  moveToThread(thread_);
  thread_->start();
}

LogReader::~LogReader() {
  // wait until thread is finished.
  exit_ = true;
  file_reader_->abort();
  thread_->quit();
  thread_->wait();

  // free all
  for (auto e : events_) delete e;
}

void LogReader::start() {
  file_reader_->read();
}

void LogReader::parseEvents(kj::ArrayPtr<const capnp::word> words) {
  QMap<int, QPair<int, int> > eidx_local;

  while (!exit_ && words.size() > 0) {
    try {
      std::unique_ptr<Event> evt = std::make_unique<Event>(words);
      words = kj::arrayPtr(evt->reader.getEnd(), words.end());

      cereal::Event::Reader event = evt->event();

      // hack
      // TODO: rewrite with callback
      if (event.which() == cereal::Event::ROAD_ENCODE_IDX) {
        auto ee = event.getRoadEncodeIdx();
        eidx_local.insert(ee.getFrameId(), qMakePair(ee.getSegmentNum(), ee.getSegmentId()));
      }
      events_.insert(event.getLogMonoTime(), evt.release());
    } catch (const kj::Exception &e) {
      // partial messages trigger this
      // qDebug() << e.getDescription().cStr();
      break;
    }
  }

  // merge in events
  // TODO: add lock
  events_lock->lockForWrite();
  *events += events_;
  eidx->unite(eidx_local);
  events_lock->unlock();
}

void LogReader::fileReady(const QByteArray &dat) {
  raw_.resize(1024 * 1024 * 64);
  if (!decompressBZ2(raw_, dat.data(), dat.size())) {
    qWarning() << "bz2 decompress failed";
  }
  parseEvents({(const capnp::word *)raw_.data(), raw_.size() / sizeof(capnp::word)});
}
