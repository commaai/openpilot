#pragma once
#include <iostream>

#include <QList>
#include <QString>
#include <QJsonArray>
#include <QJsonDocument>

#include "api.hpp"

class RouteSegment {
public:
  RouteSegment(int index_, QString log_path_, QString camera_path_) : index(index_), log_path(log_path_), camera_path(camera_path_) {}
  int index;
  QString log_path;
  QString camera_path;
};
class Route : public QWidget {
  Q_OBJECT

public:
  Route(QString route_name);
  void _get_segments_remote();

public slots:
  void parseResponse(QString response);

private:
  QString route_name;
  QList<RouteSegment*> segments;
};
