#include "selfdrive/ui/replay/filereader.h"

#include <bzlib.h>
#include <QtNetwork>
#include "cereal/gen/cpp/log.capnp.h"

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
  }
}

void FileReader::readyRead() {
  QByteArray dat = reply->readAll();
  printf("got http ready read: %d\n", dat.size());
}

FileReader::~FileReader() {

}

LogReader::LogReader(const QString &file) : FileReader(file) {
  // start with 64MB buffer
  raw_.resize(1024 * 1024 * 64);
}

LogReader::~LogReader() {
  for (auto e : events_) {
    delete e;
  }
}

void LogReader::parseEvents(kj::ArrayPtr<const capnp::word> amsg) {
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
        encoderIdx_[RoadCamFrame][ee.getFrameId()] = {ee.getSegmentNum(), ee.getSegmentId()};
      } else if (event.which() == cereal::Event::DRIVER_ENCODE_IDX) {
        auto ee = event.getDriverEncodeIdx();
        encoderIdx_[DriverCamFrame][ee.getFrameId()] = {ee.getSegmentNum(), ee.getSegmentId()};
      } else if (event.which() == cereal::Event::WIDE_ROAD_ENCODE_IDX) {
        auto ee = event.getWideRoadEncodeIdx();
        encoderIdx_[WideRoadCamFrame][ee.getFrameId()] = {ee.getSegmentNum(), ee.getSegmentId()};
      }

      events_.insert(event.getLogMonoTime(), reader.release());
    } catch (const kj::Exception &e) {
      // partial messages trigger this
      // qDebug() << e.getDescription().cStr();
      break;
    }
  }
  ready_ = true;
  emit done();
}

void LogReader::readyRead() {
  QByteArray dat = reply->readAll();
  if (!decompressBZ2(raw_, dat.data(), dat.size())) {
    qWarning() << "bz2 decompress failed";
  }
  parseEvents({(const capnp::word *)raw_.data(), raw_.size() / sizeof(capnp::word)});
}

const EncodeIdx *LogReader::getFrameEncodeIdx(FrameType type, uint32_t frame_id) const {
  auto it = encoderIdx_[type].find(frame_id);
  return it != encoderIdx_[type].end() ? &(it->second) : nullptr;
}
