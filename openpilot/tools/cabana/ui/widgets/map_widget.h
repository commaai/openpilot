#pragma once

#include <string>
#include <utility>
#include <vector>

#include "tools/cabana/streams/abstractstream.h"

namespace cabana {

class MapWidget {
public:
  MapWidget();
  void draw();
  bool visible = false;

private:
  void updatePoints();

  uint64_t last_revision_ = UINT64_MAX;
  std::vector<double> times_;
  std::vector<double> lats_;
  std::vector<double> lons_;
  double min_lat_ = 0, max_lat_ = 0;
  double min_lon_ = 0, max_lon_ = 0;
  bool bounds_set_ = false;
};

}  // namespace cabana
