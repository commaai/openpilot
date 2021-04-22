#include "replay.hpp"

Replay::Replay(QString route_, int seek, int use_api_) : route(route_), use_api(use_api_){
  unlogger = new Unlogger(&events, &events_lock, &frs, seek);
  seg_add = 0;

  if (use_api) {
    http = new HttpRequest(this, "https://api.commadotai.com/v1/route/" + route.replace("/", "|") + "/files", "HttpRequest_Test");
    QObject::connect(http, SIGNAL(receivedResponse(QString)), this, SLOT(parseResponse(QString)));
    return;
  }
  // add first segment
  addSegment(0);
}

void Replay::parseResponse(QString response){
  response = response.trimmed();
  QJsonDocument doc = QJsonDocument::fromJson(response.toUtf8());

  qDebug() << response;

	if (doc.isNull()) {
		qDebug() << "JSON Parse failed on getting past drives statistics";
		return;
	}

  camera_paths = doc["cameras"].toArray();
  log_paths = doc["logs"].toArray();

  // add first segment
  addSegment(0);
}

bool Replay::addSegment(int i){
  if (lrs.find(i) == lrs.end()) {
    QString fn = QString("http://data.comma.life/%1/%2/rlog.bz2").arg(route).arg(i);

    QThread* thread = new QThread;
    if (!use_api) {
      lrs.insert(i, new LogReader(fn, &events, &events_lock, &unlogger->eidx));
    } else {
      QString log_fn = this->log_paths.at(i).toString();
      qDebug() << log_fn;
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

void Replay::stream(int seek, SubMaster *sm){
  QThread* thread = new QThread;
  unlogger->moveToThread(thread);
  QObject::connect(thread, &QThread::started, [=](){
    unlogger->process(sm);
	});
  thread->start();

  QObject::connect(unlogger, &Unlogger::loadSegment, [=](){
    addSegment(++seg_add);
    trimSegment(seg_add > 1);
  });
}
