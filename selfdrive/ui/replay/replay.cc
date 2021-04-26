#include "replay.hpp"

Replay::Replay(QString route_, int seek) : route(route_) {
  unlogger = new Unlogger(&events, &events_lock, &frs, seek);
  current_segment = 0;
  bool create_jwt = true;

#if !defined(QCOM) && !defined(QCOM2)
  create_jwt = false;
#endif

  http = new HttpRequest(this, "https://api.commadotai.com/v1/route/" + route + "/files", "", create_jwt);
  QObject::connect(http, SIGNAL(receivedResponse(QString)), this, SLOT(parseResponse(QString)));
}

void Replay::parseResponse(QString response){
  response = response.trimmed();
  QJsonDocument doc = QJsonDocument::fromJson(response.toUtf8());

  if (doc.isNull()) {
    qDebug() << "JSON Parse failed";
    return;
  }

  camera_paths = doc["cameras"].toArray();
  log_paths = doc["logs"].toArray();

  // add first segment
  addSegment(0);
}

void Replay::addSegment(int i){
  if (lrs.find(i) != lrs.end()) {
    return;
  }

  QThread* thread = new QThread;

  QString log_fn = this->log_paths.at(i).toString();
  lrs.insert(i, new LogReader(log_fn, &events, &events_lock, &unlogger->eidx));

  lrs[i]->moveToThread(thread);
  QObject::connect(thread, SIGNAL (started()), lrs[i], SLOT (process()));
  thread->start();

  QString camera_fn = this->camera_paths.at(i).toString();
  frs.insert(i, new FrameReader(qPrintable(camera_fn)));
}

void Replay::trimSegment(int n){
  event_sizes.enqueue(events.size() - event_sizes.last());
  auto first = events.begin();

  for(int i = 0 ; i < n ; i++){
    int remove = event_sizes.dequeue();
    for(int j = 0 ; j < remove ; j++){
      first = events.erase(first);
    }
  }
}

void Replay::stream(SubMaster *sm){
  QThread* thread = new QThread;
  unlogger->moveToThread(thread);
  QObject::connect(thread, &QThread::started, [=](){
    unlogger->process(sm);
  });
  thread->start();

  QObject::connect(unlogger, &Unlogger::loadSegment, [=](){
    addSegment(++current_segment);
    if (current_segment > 1) {
      trimSegment(1);
    }
  });
}
