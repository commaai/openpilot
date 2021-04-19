#include "replay.hpp"

Replay::Replay(QString route_, int seek, int use_api_) : route(route_), use_api(use_api_){
  unlogger = new Unlogger(&events, &events_lock, &frs, seek);

  if (use_api) {
    QString settings;
    QFile file;
    file.setFileName("routes.json");
    file.open(QIODevice::ReadOnly | QIODevice::Text);
    settings = file.readAll();
    file.close();

    QJsonDocument sd = QJsonDocument::fromJson(settings.toUtf8());
    qWarning() << sd.isNull(); // <- print false :)
    QJsonObject sett2 = sd.object();

    this->camera_paths = sett2.value("camera").toArray();
    this->log_paths = sett2.value("logs").toArray();
  }
}

bool Replay::addSegment(int i){

  //unlogger->vipc_server->start_listener();

  if (lrs.find(i) == lrs.end()) {
    QString fn = QString("http://data.comma.life/%1/%2/rlog.bz2").arg(route).arg(i);

    QThread* thread = new QThread;
    if (!use_api) {
      lrs.insert(i, new LogReader(fn, &events, &events_lock, &unlogger->eidx));
    } else {
      QString log_fn = this->log_paths.at(i).toString();
      lrs.insert(i, new LogReader(log_fn, &events, &events_lock, &unlogger->eidx));
    }

    lrs[i]->moveToThread(thread);
    QObject::connect(thread, SIGNAL (started()), lrs[i], SLOT (process()));
    thread->start();

    QString frn = QString("http://data.comma.life/%1/%2/fcamera.hevc").arg(route).arg(i);

    if (!use_api) {
      frs.insert(i, new FrameReader(qPrintable(frn)));
    } else {
      QString camera_fn = this->camera_paths.at(i).toString();
      frs.insert(i, new FrameReader(qPrintable(camera_fn)));
    }
    return true;
  }
  return false;
}

void Replay::stream(int seek){
  QThread* thread = new QThread;
  unlogger->moveToThread(thread);
  QObject::connect(thread, SIGNAL (started()), unlogger, SLOT (process()));
  thread->start();

  addSegment(seek/60);
}

std::vector<std::pair<std::string, cereal::Event::Reader>> Replay::getMessages(){

  std::vector<std::pair<std::string, cereal::Event::Reader>> messages;
  for(auto i = 0 ; i < 8 ; i++){
    auto first = (events.begin()+1).key();

    for(auto e : events.values(first)){
      capnp::DynamicStruct::Reader e_ds = static_cast<capnp::DynamicStruct::Reader>(e);
      std::string type;
      KJ_IF_MAYBE(e_, e_ds.which()){
        type = e_->getProto().getName();
      }
      messages.push_back({type, e});
      if(type == "roadCameraState"){
        auto fr = e.getRoadCameraState();
        auto it = unlogger->eidx.find(fr.getFrameId());

        if(it != unlogger->eidx.end()) {
          auto pp = *it;
          if (frs.find(pp.first) != frs.end()) {
            auto frm = (frs)[pp.first];
            auto data = frm->get(pp.second);

            VisionBuf *buf = unlogger->vipc_server->get_buffer(VisionStreamType::VISION_STREAM_RGB_BACK);
            memcpy(buf->addr, data, frm->getRGBSize());
            VisionIpcBufExtra extra = {};

            unlogger->vipc_server->send(buf, &extra, false);
          }
        }
      }
    }
    events.remove(first);
  }

  return messages;
}

// What to do after Quiz:
// 1. get all events from earliest time in Events
// 2. put it in a pair with the time and the vector of this events <name, evnt>
// 3. return this and use it to update the submaster
// 4. repeat every time timerUPdate is called






