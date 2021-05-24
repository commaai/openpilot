#include "selfdrive/ui/replay/filereader.h"

#include <QtNetwork>

#include "tools/clib/framereader.h"

LogReader::LogReader(const QString& url, Events *events_, QReadWriteLock* events_lock_, QMap<int, QPair<int, int> > *eidx_) :
    url(url), events(events_), events_lock(events_lock_), eidx(eidx_) {
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

void LogReader::process() {
  HttpRequest *request = new HttpRequest(this, url, "", false);
  QObject::connect(request, &HttpRequest::receivedResponse, this, &LogReader::readyRead);
}

void LogReader::mergeEvents(int dled) {
  auto amsg = kj::arrayPtr((const capnp::word*)(raw.data() + event_offset), (dled-event_offset)/sizeof(capnp::word));
  Events events_local;
  QMap<int, QPair<int, int> > eidx_local;

  while (amsg.size() > 0) {
    try {
      capnp::FlatArrayMessageReader cmsg = capnp::FlatArrayMessageReader(amsg);

      // this needed? it is
      capnp::FlatArrayMessageReader *tmsg =
        new capnp::FlatArrayMessageReader(kj::arrayPtr(amsg.begin(), cmsg.getEnd()));

      amsg = kj::arrayPtr(cmsg.getEnd(), amsg.end());

      cereal::Event::Reader event = tmsg->getRoot<cereal::Event>();
      events_local.insert(event.getLogMonoTime(), event);

      // hack
      // TODO: rewrite with callback
      if (event.which() == cereal::Event::ROAD_ENCODE_IDX) {
        auto ee = event.getRoadEncodeIdx();
        eidx_local.insert(ee.getFrameId(), qMakePair(ee.getSegmentNum(), ee.getSegmentId()));
      }

      // increment
      event_offset = (char*)cmsg.getEnd() - raw.data();
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

void LogReader::readyRead(const QString &resp) {
  QByteArray dat = resp.toLocal8Bit();
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

