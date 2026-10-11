#pragma once

#include <string>
#include <unordered_map>
#include <vector>

struct CerealPoint {
  double t;
  double v;
};

// path (e.g. "/carState/vEgo") → sorted time series
using CerealSeriesMap = std::unordered_map<std::string, std::vector<CerealPoint>>;
