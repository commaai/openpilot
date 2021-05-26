#include "selfdrive/ui/replay/filereader.h"

#include <bzlib.h>
#include <QtNetwork>
#include "cereal/gen/cpp/log.capnp.h"

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

void FileReader::startRequest(const QUrl &url) {
  timer.start();

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
  }
}

void FileReader::readyRead() {
  QByteArray dat = reply->readAll();
  emit ready(dat);
}

LogReader::LogReader(const QString &file, QObject *parent) : file_(file), QThread(parent) {}

LogReader::~LogReader() {
  // wait until thread is finished.
  exit_ = true;
  wait();
  for (auto e : events_) delete e;
}

void LogReader::run() {
  QEventLoop loop;
  FileReader reader;
  connect(&reader, &FileReader::ready, [&](const QByteArray &dat) {
    if (!decompressBZ2(raw_, dat.data(), dat.size())) {
      qWarning() << "bz2 decompress failed";
    }
    loop.exit();
  });
  reader.startRequest(file_);
  loop.exec();

  parseEvents({(const capnp::word *)raw_.data(), raw_.size() / sizeof(capnp::word)});
}

void LogReader::parseEvents(kj::ArrayPtr<const capnp::word> words) {
  auto insertEidx = [=](FrameType type, const cereal::EncodeIndex::Reader &e) {
    encoderIdx_[type][e.getFrameId()] = {e.getSegmentNum(), e.getSegmentId()};
  };

  bool success = true;
  while (!exit_ && words.size() > 0) {
    try {
      std::unique_ptr<capnp::FlatArrayMessageReader> message = std::make_unique<capnp::FlatArrayMessageReader>(words);
      cereal::Event::Reader event = message->getRoot<cereal::Event>();
      switch (event.which()) {
        case cereal::Event::ROAD_ENCODE_IDX:
          insertEidx(RoadCamFrame, event.getRoadEncodeIdx());
          break;
        case cereal::Event::DRIVER_ENCODE_IDX:
          insertEidx(DriverCamFrame, event.getDriverEncodeIdx());
          break;
        case cereal::Event::WIDE_ROAD_ENCODE_IDX:
          insertEidx(WideRoadCamFrame, event.getWideRoadEncodeIdx());
          break;
        default:
          break;
      }
      events_.insert(event.getLogMonoTime(), message.release());
      words = kj::arrayPtr(message->getEnd(), words.end());
    } catch (const kj::Exception &e) {
      // partial messages trigger this
      // qDebug() << e.getDescription().cStr();
      success = false;
      break;
    }
  }
  emit finished(success && !exit_);
}

void LogReader::readyRead(const QByteArray &dat) {
  // start with 64MB buffer
  raw_.resize(1024 * 1024 * 64);
  if (!decompressBZ2(raw_, dat.data(), dat.size())) {
    qWarning() << "bz2 decompress failed";
  }
  parseEvents({(const capnp::word *)raw_.data(), raw_.size() / sizeof(capnp::word)});
}
