#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "tools/cabana/streams/cerealseries.h"
#include "json11/json11.hpp"

namespace cabana {
struct PythonSeries {
  std::string linked_source;
  std::vector<std::string> additional_sources;
  std::string globals_code;
  std::string function_code;
};
struct LayoutCurve {
  std::string name;
  std::string can_id;
  std::string color;
  bool visible = true;
  SeriesTransform transform;
  std::optional<PythonSeries> python;
};
struct LayoutChart {
  std::string title;
  int series_type = 0;
  std::optional<std::pair<double, double>> y_limits;
  std::vector<LayoutCurve> curves;
};
struct LayoutTab {
  std::string name;
  std::vector<LayoutChart> charts;
};
struct AnalysisLayout {
  std::vector<LayoutTab> tabs;
  int current_tab = 0;
};
AnalysisLayout parseAnalysisLayout(const std::string &text);
std::string writeAnalysisLayout(const AnalysisLayout &layout);
std::string pythonSeriesKey(const PythonSeries &series);
CerealPoints evaluatePythonSeries(const PythonSeries &spec, const std::map<std::string, CerealPoints> &inputs,
                                 const CerealCancel &cancelled, double timeout_seconds = 15.0, uint64_t time_origin = 0);
}  // namespace cabana
