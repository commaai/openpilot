#include "route.hpp"

#include <QDebug>

Route::Route(QString route_name_) : route_name(route_name_){
	_get_segments_remote();
}

void Route::_get_segments_remote(){
	segments.clear();
  QString url = QString("https://api.commadotai.com/v1/route/" + route_name + "/files");

  RequestRepeater *repeater = new RequestRepeater(nullptr, url, 2, "ApiCache_Route");
  QObject::connect(repeater, SIGNAL(receivedResponse(QString)), this, SLOT(parseResponse(QString)));
}

void Route::parseResponse(QString response){
  response = response.trimmed();
  QJsonDocument doc = QJsonDocument::fromJson(response.toUtf8());

  if (doc.isNull()) {
    qDebug() << "JSON Parse failed on getting past drives statistics";
    return;
  }

  auto cams = doc["cameras"].toArray();
  auto logs = doc["logs"].toArray();

  for(int i = 0 ; i < cams.size() ; i++){
    segments.append(new RouteSegment(i, logs[i].toString(), cams[i].toString()));
  }

	emit doneParsing();
}

QList<QString> Route::log_paths() {
  QList<QString> paths;
  for(auto &seg : segments){
    paths.append(seg->log_path);
  }
  return paths;
}

QList<QString> Route::camera_paths() {
  QList<QString> paths;
  for(auto &seg : segments){
    paths.append(seg->camera_path);
  }
  return paths;
}
