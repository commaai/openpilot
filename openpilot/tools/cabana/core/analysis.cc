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
                              {"globals_code", spec.globals_code}, {"function_code", spec.function_code}, {"language", spec.language}};
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
  curve.label = node["label"].string_value();
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
  curve.transform.time_offset = number(node, "time_offset", 0.0);
  if (curve.transform.derivative_dt < 0) throw std::invalid_argument("Negative derivative divisor");
  if (node["custom_python"].is_object()) {
    const auto &custom = node["custom_python"];
    PythonSeries spec;
    spec.linked_source = pathName(custom["linked_source"].string_value());
    spec.globals_code = custom["globals_code"].string_value();
    spec.function_code = custom["function_code"].string_value();
    if (custom["language"].is_string()) spec.language = custom["language"].string_value();
    if (spec.language != "python" && spec.language != "lua") throw std::invalid_argument("Unsupported calculation language");
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
LayoutNode parsePanes(const json11::Json &node, std::vector<LayoutChart> &charts, int depth = 0) {
  if (depth > 64) throw std::invalid_argument("Layout nesting exceeds 64 levels");
  LayoutNode tree;
  if (!node.is_object()) throw std::invalid_argument("Invalid layout pane");
  const auto kind = node["kind"].string_value();
  if (!kind.empty() && kind != "plot" && kind != "map" && kind != "camera" && kind != "thumbnail") {
    throw std::invalid_argument("Unsupported pane kind: " + kind);
  }
  if (node["curves"].is_array() || !kind.empty() || node.object_items().empty()) {
    LayoutChart chart;
    chart.title = node["title"].string_value();
    chart.kind = kind.empty() ? "plot" : kind;
    chart.flip_x = node["flip_x"].bool_value(); chart.flip_y = node["flip_y"].bool_value();
    chart.camera_view = node["camera_view"].is_string() ? node["camera_view"].string_value() : "road";
    if (chart.camera_view == "driver") chart.camera_view = "cabin";
    if (chart.camera_view != "road" && chart.camera_view != "cabin" && chart.camera_view != "wide_road" && chart.camera_view != "qroad") {
      throw std::invalid_argument("Unknown camera view: " + chart.camera_view);
    }
    if (node["x_limits"].is_object() || node["range"].is_object()) {
      const auto &range = node["x_limits"].is_object() ? node["x_limits"] : node["range"];
      const double left = number(range, "left", 0), right = number(range, "right", 1);
      if (left >= right) throw std::invalid_argument("Invalid X range");
      chart.x_limits = {left, right};
    }
    const double series_type = number(node, "series_type", 0);
    if (series_type < 0 || series_type > 2 || std::floor(series_type) != series_type) {
      throw std::invalid_argument("Unsupported chart type");
    }
    chart.series_type = static_cast<int>(series_type);
    if (node["y_limits"].is_object()) {
      const double min = number(node["y_limits"], "min", number(node["range"], "bottom", 0));
      const double max = number(node["y_limits"], "max", number(node["range"], "top", 1));
      if (min >= max) throw std::invalid_argument("Invalid Y limits");
      chart.y_limits = {min, max};
    } else if (node["range"].is_object()) {
      const double min = number(node["range"], "bottom", 0);
      const double max = number(node["range"], "top", 1);
      if (min >= max) throw std::invalid_argument("Invalid Y range");
      chart.y_limits = {min, max};
    }
    for (const auto &curve : node["curves"].array_items()) chart.curves.push_back(parseCurve(curve));
    tree.chart = charts.size();
    charts.push_back(std::move(chart));
  } else if (node["children"].is_array()) {
    const auto split = node["split"].string_value();
    if (split != "horizontal" && split != "vertical") throw std::invalid_argument("Invalid layout split");
    tree.horizontal = split == "horizontal";
    if (node["children"].array_items().empty()) throw std::invalid_argument("Empty layout split");
    for (const auto &child : node["children"].array_items()) tree.children.push_back(parsePanes(child, charts, depth + 1));
    if (!node["sizes"].is_null() && (!node["sizes"].is_array() || node["sizes"].array_items().size() != tree.children.size())) {
      throw std::invalid_argument("Split sizes do not match its children");
    }
    for (const auto &size : node["sizes"].array_items()) {
      if (!size.is_number() || !std::isfinite(size.number_value()) || size.number_value() <= 0) throw std::invalid_argument("Invalid split size");
      tree.sizes.push_back(size.number_value());
    }
    if (tree.sizes.empty()) tree.sizes.resize(tree.children.size(), 1.0);
  } else {
    throw std::invalid_argument("Layout pane has neither curves nor children");
  }
  return tree;
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
  if (!doc["calculations"].is_null() && !doc["calculations"].is_object()) throw std::invalid_argument("Invalid calculation registry");
  for (const auto &[name, spec] : doc["calculations"].object_items()) {
    if (!spec.is_object()) throw std::invalid_argument("Invalid calculation definition: " + name);
    layout.calculations[name] = *parseCurve(json11::Json::object{{"name", name}, {"custom_python", spec}}).python;
  }
  layout.current_tab = doc["current_tab_index"].int_value();
  for (const auto &tab : doc["tabs"].array_items()) {
    LayoutTab result;
    result.name = tab["name"].string_value();
    if (tab["charts"].is_array()) {
      for (const auto &chart : tab["charts"].array_items()) parsePanes(chart, result.charts);
    } else {
      result.root = parsePanes(tab["root"], result.charts);
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
        json11::Json::object node{{"name", curve.name}, {"label", curve.label}, {"color", curve.color}, {"visible", curve.visible},
                                  {"scale", curve.transform.scale}, {"offset", curve.transform.offset},
                                  {"derivative_dt", curve.transform.derivative_dt}, {"time_offset", curve.transform.time_offset}};
        if (!curve.can_id.empty()) node["can_id"] = curve.can_id;
        if (curve.transform.derivative) node["transform"] = "derivative";
        if (curve.python) node["custom_python"] = pythonJson(*curve.python);
        curves.push_back(node);
      }
      json11::Json::object node{{"title", chart.title}, {"kind", chart.kind}, {"camera_view", chart.camera_view},
                                {"series_type", chart.series_type}, {"flip_x", chart.flip_x}, {"flip_y", chart.flip_y}, {"curves", curves}};
      if (chart.x_limits) node["x_limits"] = json11::Json::object{{"left", chart.x_limits->first}, {"right", chart.x_limits->second}};
      if (chart.y_limits) node["y_limits"] = json11::Json::object{{"min", chart.y_limits->first}, {"max", chart.y_limits->second}};
      charts.push_back(node);
    }
    if (tab.root) {
      std::function<json11::Json(const LayoutNode &, int)> encode = [&](const LayoutNode &node, int depth) -> json11::Json {
        if (depth > 64) throw std::invalid_argument("Layout nesting exceeds 64 levels");
        if (node.chart >= 0) {
          if (node.chart >= static_cast<int>(charts.size())) throw std::invalid_argument("Invalid layout pane index");
          return charts[node.chart];
        }
        json11::Json::array children, sizes;
        for (const auto &child : node.children) children.push_back(encode(child, depth + 1));
        for (double size : node.sizes) sizes.push_back(size);
        return json11::Json::object{{"split", node.horizontal ? "horizontal" : "vertical"}, {"sizes", sizes}, {"children", children}};
      };
      tabs.push_back(json11::Json::object{{"name", tab.name}, {"root", encode(*tab.root, 0)}});
    } else tabs.push_back(json11::Json::object{{"name", tab.name}, {"charts", charts}});
  }
  json11::Json::object calculations;
  for (const auto &[name, spec] : layout.calculations) calculations[name] = pythonJson(spec);
  return json11::Json(json11::Json::object{{"cabana_layout", 1}, {"current_tab_index", layout.current_tab}, {"tabs", tabs}, {"calculations", calculations}}).dump();
}
}  // namespace cabana
