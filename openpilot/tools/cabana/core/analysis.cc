#include "tools/cabana/core/analysis.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <stdexcept>

namespace cabana {
namespace {
std::string pathName(std::string path) {
  if (!path.empty() && path.front() == '/') path.erase(0, 1);
  return path;
}
json11::Json pythonJson(const PythonSeries &spec) {
  json11::Json::array additional;
  for (const auto &path : spec.additional_sources) additional.push_back(path);
  return json11::Json::object{{"linked_source", spec.linked_source}, {"additional_sources", additional},
                              {"globals_code", spec.globals_code}, {"function_code", spec.function_code}};
}
double number(const json11::Json &node, const char *key, double fallback) {
  auto value = node[key];
  if (value.is_null()) return fallback;
  if (!value.is_number() || !std::isfinite(value.number_value())) {
    throw std::invalid_argument(std::string("Invalid layout number: ") + key);
  }
  return value.number_value();
}
LayoutCurve parseCurve(const json11::Json &node) {
  if (!node.is_object()) throw std::invalid_argument("A layout curve must be an object");
  LayoutCurve curve;
  curve.name = node["name"].string_value();
  curve.can_id = node["can_id"].string_value();
  curve.color = node["color"].string_value();
  if (curve.name.empty()) throw std::invalid_argument("A layout curve has no name");
  if (!curve.color.empty() && ((curve.color.size() != 7 && curve.color.size() != 9) || curve.color.front() != '#' ||
      !std::all_of(curve.color.begin() + 1, curve.color.end(), [](unsigned char c) { return std::isxdigit(c); }))) {
    throw std::invalid_argument("Invalid curve color: " + curve.color);
  }
  curve.visible = !node["visible"].is_bool() || node["visible"].bool_value();
  const auto transform = node["transform"].string_value();
  if (!transform.empty() && transform != "derivative" && transform != "scale") {
    throw std::invalid_argument("Unsupported transform: " + transform);
  }
  curve.transform.derivative = transform == "derivative";
  curve.transform.derivative_dt = number(node, "derivative_dt", 0.0);
  curve.transform.scale = number(node, "scale", 1.0);
  curve.transform.offset = number(node, "offset", 0.0);
  if (curve.transform.derivative_dt < 0) throw std::invalid_argument("Negative derivative divisor");
  if (node["custom_python"].is_object()) {
    const auto &custom = node["custom_python"];
    PythonSeries spec;
    spec.linked_source = pathName(custom["linked_source"].string_value());
    spec.globals_code = custom["globals_code"].string_value();
    spec.function_code = custom["function_code"].string_value();
    for (const auto &source : custom["additional_sources"].array_items()) {
      if (!source.is_string() || source.string_value().empty()) throw std::invalid_argument("Invalid Python input path");
      spec.additional_sources.push_back(pathName(source.string_value()));
    }
    if (spec.linked_source.empty() || spec.function_code.empty()) throw std::invalid_argument("Python curve is missing its source or code");
    curve.python = std::move(spec);
  } else if (curve.can_id.empty() && curve.name.front() != '/') {
    throw std::invalid_argument("Unresolved derived signal: " + curve.name);
  }
  return curve;
}
void parsePanes(const json11::Json &node, std::vector<LayoutChart> &charts) {
  if (!node.is_object()) throw std::invalid_argument("Invalid layout pane");
  const auto kind = node["kind"].string_value();
  if (!kind.empty() && kind != "plot") throw std::invalid_argument("This layout requires a " + kind + " pane");
  if (node["curves"].is_array()) {
    LayoutChart chart;
    chart.title = node["title"].string_value();
    const double series_type = number(node, "series_type", 0);
    if (series_type < 0 || series_type > 2 || std::floor(series_type) != series_type) {
      throw std::invalid_argument("Unsupported chart type");
    }
    chart.series_type = static_cast<int>(series_type);
    if (node["y_limits"].is_object()) {
      const double min = number(node["y_limits"], "min", 0);
      const double max = number(node["y_limits"], "max", 1);
      if (min >= max) throw std::invalid_argument("Invalid Y limits");
      chart.y_limits = {min, max};
    } else if (node["range"].is_object()) {
      const double min = number(node["range"], "bottom", 0);
      const double max = number(node["range"], "top", 1);
      if (min >= max) throw std::invalid_argument("Invalid Y range");
      chart.y_limits = {min, max};
    }
    for (const auto &curve : node["curves"].array_items()) chart.curves.push_back(parseCurve(curve));
    charts.push_back(std::move(chart));
  } else if (node["children"].is_array()) {
    for (const auto &child : node["children"].array_items()) parsePanes(child, charts);
  } else {
    throw std::invalid_argument("Layout pane has neither curves nor children");
  }
}
}  // namespace

AnalysisLayout parseAnalysisLayout(const std::string &text) {
  std::string error;
  auto doc = json11::Json::parse(text, error);
  if (!error.empty() || !doc.is_object()) throw std::invalid_argument("Invalid layout JSON: " + error);
  if (!doc["cabana_layout"].is_null() && doc["cabana_layout"].int_value() != 1) {
    throw std::invalid_argument("Unsupported Cabana layout version");
  }
  if (!doc["tabs"].is_array() || doc["tabs"].array_items().empty()) throw std::invalid_argument("Layout has no tabs");
  AnalysisLayout layout;
  layout.current_tab = doc["current_tab_index"].int_value();
  for (const auto &tab : doc["tabs"].array_items()) {
    LayoutTab result;
    result.name = tab["name"].string_value();
    if (tab["charts"].is_array()) {
      for (const auto &chart : tab["charts"].array_items()) parsePanes(chart, result.charts);
    } else {
      parsePanes(tab["root"], result.charts);
    }
    layout.tabs.push_back(std::move(result));
  }
  if (layout.current_tab < 0 || layout.current_tab >= static_cast<int>(layout.tabs.size())) {
    throw std::invalid_argument("Layout selects a nonexistent tab");
  }
  return layout;
}

std::string pythonSeriesKey(const PythonSeries &spec) { return "python:" + pythonJson(spec).dump(); }

std::string writeAnalysisLayout(const AnalysisLayout &layout) {
  json11::Json::array tabs;
  for (const auto &tab : layout.tabs) {
    json11::Json::array charts;
    for (const auto &chart : tab.charts) {
      json11::Json::array curves;
      for (const auto &curve : chart.curves) {
        json11::Json::object node{{"name", curve.name}, {"color", curve.color}, {"visible", curve.visible},
                                  {"scale", curve.transform.scale}, {"offset", curve.transform.offset},
                                  {"derivative_dt", curve.transform.derivative_dt}};
        if (!curve.can_id.empty()) node["can_id"] = curve.can_id;
        if (curve.transform.derivative) node["transform"] = "derivative";
        if (curve.python) node["custom_python"] = pythonJson(*curve.python);
        curves.push_back(node);
      }
      json11::Json::object node{{"title", chart.title}, {"series_type", chart.series_type}, {"curves", curves}};
      if (chart.y_limits) node["y_limits"] = json11::Json::object{{"min", chart.y_limits->first}, {"max", chart.y_limits->second}};
      charts.push_back(node);
    }
    tabs.push_back(json11::Json::object{{"name", tab.name}, {"charts", charts}});
  }
  return json11::Json(json11::Json::object{{"cabana_layout", 1}, {"current_tab_index", layout.current_tab}, {"tabs", tabs}}).dump();
}
}  // namespace cabana
