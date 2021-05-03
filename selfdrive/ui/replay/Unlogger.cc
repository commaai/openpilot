#include <cmath>
#include <string>
#include <vector>
#include <capnp/dynamic.h>
#include <capnp/schema.h>

// include the dynamic struct
#include "cereal/gen/cpp/log.capnp.c++"
#include "cereal/gen/cpp/car.capnp.c++"
#include "cereal/gen/cpp/legacy.capnp.c++"
#include "cereal/services.h"

#include "Unlogger.h"

#include <stdint.h>
#include <time.h>

#include "common/timing.h"

Unlogger::Unlogger(Events *events_, QReadWriteLock* events_lock_, QMap<int, FrameReader*> *frs_, int seek)
  : events(events_), events_lock(events_lock_), frs(frs_) {
  ctx = Context::create();

  seek_request = seek*1e9;

  QStringList block = QString(getenv("BLOCK")).split(",");
  qDebug() << "blocklist" << block;

  QStringList allow = QString(getenv("ALLOW")).split(",");
  qDebug() << "allowlist" << allow;

  for (const auto& it : services) {
    std::string name = it.name;
    if (allow[0].size() > 0 && !allow.contains(name.c_str())) {
      qDebug() << "not allowing" << name.c_str();
      continue;
    }

    if (block.contains(name.c_str())) {
      qDebug() << "blocking" << name.c_str();
      continue;
    }

    PubSocket *sock = PubSocket::create(ctx, name);
    if (sock == NULL) {
      qDebug() << "FAILED" << name.c_str();
      continue;
    }

    qDebug() << name.c_str();

    socks.insert(name, sock);
  }
}

void Unlogger::process(SubMaster *sm) {

  qDebug() << "hello from unlogger thread";
  while (events->size() == 0) {
    qDebug() << "waiting for events";
    QThread::sleep(1);
  }
  qDebug() << "got events";

  // TODO: hack
  if (seek_request != 0) {
    seek_request += events->begin().key();
    while (events->lowerBound(seek_request) == events->end()) {
      qDebug() << "waiting for desired time";
      QThread::sleep(1);
    }
  }

  QElapsedTimer timer;
  timer.start();

  uint64_t last_elapsed = 0;

  // loops
  while (1) {
    uint64_t t0 = (events->begin()+1).key();
    uint64_t t0r = timer.nsecsElapsed();
    qDebug() << "unlogging at" << t0;

    auto eit = events->lowerBound(t0);
    while (eit != events->end()) {

      float time_to_end = ((events->lastKey() - eit.key())/1e9);
      if (loading_segment && (time_to_end > 20.0)){
        loading_segment = false;
      }

      while (paused) {
        QThread::usleep(1000);
        t0 = eit->getLogMonoTime();
        t0r = timer.nsecsElapsed();
      }

      if (seek_request != 0) {
        t0 = seek_request;
        qDebug() << "seeking to" << t0;
        t0r = timer.nsecsElapsed();
        eit = events->lowerBound(t0);
        seek_request = 0;
        if (eit == events->end()) {
          qWarning() << "seek off end";
          break;
        }
      }

      if (abs(((long long)tc-(long long)last_elapsed)) > 50e6) {
        //qDebug() << "elapsed";
        emit elapsed();
        last_elapsed = tc;
      }

      cereal::Event::Reader e = *eit;

      capnp::DynamicStruct::Reader e_ds = static_cast<capnp::DynamicStruct::Reader>(e);
      std::string type;
      KJ_IF_MAYBE(e_, e_ds.which()){
        type = e_->getProto().getName();
      }

      uint64_t tm = e.getLogMonoTime();
      auto it = socks.find(type);
      tc = tm;
      if (it != socks.end()) {
        long etime = tm-t0;

        float timestamp = etime/1e9;
        if(std::abs(timestamp-last_print) > 5.0){
          last_print = timestamp;
          printf("at %f\n", last_print);
        }

        long rtime = timer.nsecsElapsed() - t0r;
        long us_behind = ((etime-rtime)*1e-3)+0.5;
        if (us_behind > 0) {
          if (us_behind > 1e6) {
            qWarning() << "OVER ONE SECOND BEHIND, HACKING" << us_behind;
            us_behind = 0;
            t0 = tm;
            t0r = timer.nsecsElapsed();
          }
          QThread::usleep(us_behind);
          //qDebug() << "sleeping" << us_behind << etime << timer.nsecsElapsed();
        }

        if (type == "roadCameraState") {
          auto fr = e.getRoadCameraState();

          auto it_ = eidx.find(fr.getFrameId());
          if (it_ != eidx.end()) {
            auto pp = *it_;
            //qDebug() << fr.getRoadCameraStateId() << pp;

            if (frs->find(pp.first) != frs->end()) {
              auto frm = (*frs)[pp.first];
              auto data = frm->get(pp.second);

              if (vipc_server == nullptr) {
                cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
                cl_context context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));

                vipc_server = new VisionIpcServer("camerad", device_id, context);
                vipc_server->create_buffers(VisionStreamType::VISION_STREAM_RGB_BACK, 4, true, frm->width, frm->height);

                vipc_server->start_listener();
              }

              VisionBuf *buf = vipc_server->get_buffer(VisionStreamType::VISION_STREAM_RGB_BACK);
              memcpy(buf->addr, data, frm->getRGBSize());
              VisionIpcBufExtra extra = {};

              vipc_server->send(buf, &extra, false);
            }
          }
        }

        if (sm == nullptr){
          capnp::MallocMessageBuilder msg;
          msg.setRoot(e);
          auto words = capnp::messageToFlatArray(msg);
          auto bytes = words.asBytes();

          (*it)->send((char*)bytes.begin(), bytes.size());
        } else{
          std::vector<std::pair<std::string, cereal::Event::Reader>> messages;
          messages.push_back({type, e});
          sm->update_msgs(nanos_since_boot(), messages);
        }
      }
      ++eit;

      if (time_to_end < 10.0 && !loading_segment){
        loading_segment = true;
        emit loadSegment();
      }
    }
  }
}
