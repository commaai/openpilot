#pragma once
#include <iostream>

#include <QString>
#include <QJsonArray>
#include <QJsonDocument>

#include "api.hpp"

class Route : public QWidget {
  Q_OBJECT

public:
  Route(QString route_name);
  void _get_segments_remote();

public slots:
  void parseResponse(QString response);

private:
  QString route_name;
};
