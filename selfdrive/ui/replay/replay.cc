#include "replay.hpp"

Replay::Replay(QString route_, int seek) : route(route_) {
  unlogger = new Unlogger(&events, &events_lock, &frs, seek);
  current_segment = 0;
  bool create_jwt = true;

#if !defined(QCOM) && !defined(QCOM2)
  create_jwt = false;
#endif

  http = new HttpRequest(this, "https://api.commadotai.com/v1/route/" + route + "/files", "", create_jwt);
  QObject::connect(http, &HttpRequest::receivedResponse, this, &Replay::parseResponse);
}

void Replay::parseResponse(const QString &response){
  QJsonDocument doc = QJsonDocument::fromJson(response.trimmed().toUtf8());

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
  QObject::connect(thread, &QThread::started, lrs[i], &LogReader::process);
  thread->start();

  QString camera_fn = this->camera_paths.at(i).toString();
  frs.insert(i, new FrameReader(qPrintable(camera_fn)));
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
  });
}
