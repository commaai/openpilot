#pragma once

#include <QString>
#include <iostream>

#include "api.hpp"

class Route {

public:
  Route(QString route_name);

  void _get_segments_remote();


private:
  QString route_name;
};
