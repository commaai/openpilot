#include "selfdrive/ui/replay/filereader.h"

#include <bzlib.h>
#include <QtNetwork>

#include "selfdrive/common/timing.h"

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

LogReader::LogReader(const QString &file, QObject *parent) : QObject(parent) {
  thread_ = new QThread();
  moveToThread(thread_);
  connect(thread_, &QThread::started, this, &LogReader::start);
  
  file_reader_ = new FileReader(file);
  connect(file_reader_, &FileReader::finished, this, &LogReader::parseEvents);
  connect(file_reader_, &FileReader::failed, [=](const QString &err) { qInfo() << err; });

  thread_->start();
  qDebug() << "read log " << file;
}

LogReader::~LogReader() {
  // wait until thread is finished.
  exit_ = true;
  file_reader_->abort();
  thread_->quit();
  thread_->wait();
  thread_->deleteLater();
}

void LogReader::start() {
  file_reader_->read();
}

void LogReader::parseEvents(const QByteArray &dat) {
  raw_.resize(1024 * 1024 * 64);
  if (!decompressBZ2(raw_, dat.data(), dat.size())) {
    qWarning() << "bz2 decompress failed";
  }

  auto insertEidx = [&](CameraType type, const cereal::EncodeIndex::Reader &e) {
    encoderIdx[type][e.getFrameId()] = {e.getSegmentNum(), e.getSegmentId()};
  };

  valid_ = true;
  kj::ArrayPtr<const capnp::word> words((const capnp::word *)raw_.data(), raw_.size() / sizeof(capnp::word));
  while (!exit_ && words.size() > 0) {
    try {
      std::unique_ptr<Event> evt = std::make_unique<Event>(words);
      switch (evt->event.which()) {
        case cereal::Event::ROAD_ENCODE_IDX:
          insertEidx(RoadCam, evt->event.getRoadEncodeIdx());
          break;
        case cereal::Event::DRIVER_ENCODE_IDX:
          insertEidx(DriverCam, evt->event.getDriverEncodeIdx());
          break;
        case cereal::Event::WIDE_ROAD_ENCODE_IDX:
          insertEidx(WideRoadCam, evt->event.getWideRoadEncodeIdx());
          break;
        default:
          break;
      }
      words = kj::arrayPtr(evt->reader.getEnd(), words.end());
      events.push_back(evt.release());
    } catch (const kj::Exception &e) {
      // partial messages trigger this
      // qDebug() << e.getDescription().cStr();
      valid_ = false;
      break;
    }
  }
  if (!exit_) {
    std::sort(events.begin(), events.end(), [=](const Event* l, const Event*r) {
      return *l < *r;
    });
    emit finished(valid_);  
  }
}
