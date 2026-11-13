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
  std::string language = "python";
};
struct LayoutCurve {
  std::string name;
  std::string label;
  std::string can_id;
  std::string color;
  bool visible = true;
  SeriesTransform transform;
  std::optional<PythonSeries> python;
};
struct LayoutChart {
  std::string title;
  std::string kind = "plot";
  std::string camera_view = "road";
  std::optional<std::pair<double, double>> x_limits;
  int series_type = 0;
  bool flip_x = false, flip_y = false;
  std::optional<std::pair<double, double>> y_limits;
  std::vector<LayoutCurve> curves;
};
struct LayoutNode {
  int chart = -1;
  bool horizontal = false;
  std::vector<double> sizes;
  std::vector<LayoutNode> children;
};
struct LayoutTab {
  std::string name;
  std::vector<LayoutChart> charts;
  std::optional<LayoutNode> root;
};
struct AnalysisLayout {
  std::map<std::string, PythonSeries> calculations;
  std::vector<LayoutTab> tabs;
  int current_tab = 0;
};
std::string convertLegacyLayout(const std::string &path);
AnalysisLayout parseAnalysisLayout(const std::string &text);
std::string writeAnalysisLayout(const AnalysisLayout &layout);
std::string pythonSeriesKey(const PythonSeries &series);
CerealPoints evaluatePythonSeries(const PythonSeries &spec, const std::map<std::string, CerealPoints> &inputs,
                                 const CerealCancel &cancelled, double timeout_seconds = 15.0, uint64_t time_origin = 0);
}  // namespace cabana
