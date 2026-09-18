#include "tools/cabana/ui/layout_manager.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "json11/json11.hpp"
#include "tools/cabana/utils/util.h"

namespace fs = std::filesystem;

namespace cabana {
namespace {

LayoutCurve parseCurveNode(const json11::Json &node) {
  LayoutCurve curve;
  curve.name = node["name"].string_value();
  curve.color_hex = node["color"].string_value();
  curve.can_id = node["can_id"].string_value();
  curve.visible = !node["visible"].is_bool() || node["visible"].bool_value();

  const std::string transform = node["transform"].string_value();
  if (transform == "derivative") {
    curve.derivative = true;
    if (node["derivative_dt"].is_number()) {
      curve.derivative_dt = node["derivative_dt"].number_value();
    }
  } else if (transform == "scale") {
    if (node["scale"].is_number()) curve.scale = node["scale"].number_value();
    if (node["offset"].is_number()) curve.offset = node["offset"].number_value();
  }

  const auto &custom = node["custom_python"];
  if (custom.is_object()) {
    CustomPythonSeries spec;
    spec.linked_source = custom["linked_source"].string_value();
    spec.globals_code = custom["globals_code"].string_value();
    spec.function_code = custom["function_code"].string_value();
    for (const auto &src : custom["additional_sources"].array_items()) {
      if (src.is_string()) spec.additional_sources.push_back(src.string_value());
    }
    curve.custom_python = std::move(spec);
  }
  return curve;
}

void parsePanesRecursive(const json11::Json &node, std::vector<LayoutPane> &panes) {
  if (!node.is_object()) return;

  if (node["curves"].is_array()) {
    LayoutPane pane;
    pane.title = node["title"].string_value();
    pane.series_type = std::clamp(node["series_type"].int_value(), 0, 2);
    pane.kind = node["kind"].string_value();
    if (node["y_limits"].is_object()) {
      const auto &lim = node["y_limits"];
      double y_min = lim["min"].is_number() ? lim["min"].number_value() : -1.0;
      double y_max = lim["max"].is_number() ? lim["max"].number_value() : 1.0;
      pane.y_limits = std::make_pair(y_min, y_max);
    }
    for (const auto &c_node : node["curves"].array_items()) {
      if (c_node.is_object()) {
        pane.curves.push_back(parseCurveNode(c_node));
      }
    }
    panes.push_back(std::move(pane));
    return;
  }

  if (node["children"].is_array()) {
    for (const auto &child : node["children"].array_items()) {
      parsePanesRecursive(child, panes);
    }
  }
}

LayoutTab parseTabNode(const json11::Json &tab_node) {
  LayoutTab tab;
  tab.name = tab_node["name"].string_value();
  if (tab.name.empty()) tab.name = "Tab";
  parsePanesRecursive(tab_node["root"], tab.panes);
  return tab;
}

json11::Json curveToJson(const LayoutCurve &curve) {
  json11::Json::object obj = {
    {"name", curve.name},
    {"color", curve.color_hex.empty() ? "#0072b2" : curve.color_hex}
  };
  obj["visible"] = curve.visible;
  if (!curve.can_id.empty()) obj["can_id"] = curve.can_id;
  if (curve.derivative) {
    obj["transform"] = "derivative";
    if (curve.derivative_dt > 0) obj["derivative_dt"] = curve.derivative_dt;
  } else if (curve.scale != 1.0 || curve.offset != 0.0) {
    obj["transform"] = "scale";
    obj["scale"] = curve.scale;
    obj["offset"] = curve.offset;
  }
  if (curve.custom_python.has_value()) {
    json11::Json::array additional;
    for (const auto &s : curve.custom_python->additional_sources) additional.push_back(s);
    obj["custom_python"] = json11::Json::object{
      {"linked_source", curve.custom_python->linked_source},
      {"additional_sources", additional},
      {"globals_code", curve.custom_python->globals_code},
      {"function_code", curve.custom_python->function_code}
    };
  }
  return obj;
}

fs::path findLayoutsDirectory() {
  fs::path direct = executableDir() / "layouts";
  if (fs::exists(direct)) return direct;
  fs::path repo_rel = executableDir() / "openpilot" / "tools" / "cabana" / "layouts";
  if (fs::exists(repo_rel)) return repo_rel;
  return direct;
}

}  // namespace

Layout LayoutManager::loadLayout(const fs::path &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Failed to open layout file: " + path.string());
  std::stringstream ss;
  ss << in.rdbuf();
  std::string err;
  const json11::Json root = json11::Json::parse(ss.str(), err);
  if (!err.empty() || !root.is_object()) {
    throw std::runtime_error("Failed to parse layout JSON: " + err);
  }

  Layout layout;
  layout.name = path.stem().string();
  layout.current_tab_index = root["current_tab_index"].is_number() ? root["current_tab_index"].int_value() : 0;
  for (const auto &tab_node : root["tabs"].array_items()) {
    if (tab_node.is_object()) {
      layout.tabs.push_back(parseTabNode(tab_node));
    }
  }
  if (layout.tabs.empty()) {
    throw std::runtime_error("Layout has no tabs: " + path.string());
  }
  layout.current_tab_index = std::clamp(layout.current_tab_index, 0, (int)layout.tabs.size() - 1);
  return layout;
}

bool LayoutManager::saveLayout(const Layout &layout, const fs::path &path) {
  json11::Json::array tabs_json;
  for (const auto &tab : layout.tabs) {
    json11::Json::array children;
    for (const auto &pane : tab.panes) {
      json11::Json::array curves;
      for (const auto &curve : pane.curves) {
        curves.push_back(curveToJson(curve));
      }
      json11::Json::object pane_obj = {
        {"title", pane.title.empty() ? "..." : pane.title},
        {"curves", curves},
        {"series_type", pane.series_type}
      };
      if (!pane.kind.empty()) pane_obj["kind"] = pane.kind;
      if (pane.y_limits.has_value()) {
        pane_obj["y_limits"] = json11::Json::object{
          {"min", pane.y_limits->first},
          {"max", pane.y_limits->second}
        };
      }
      children.push_back(pane_obj);
    }
    json11::Json::array sizes(children.size(), children.empty() ? 1.0 : 1.0 / children.size());
    json11::Json::object root_node = {
      {"split", "vertical"},
      {"sizes", sizes},
      {"children", children}
    };
    tabs_json.push_back(json11::Json::object{
      {"name", tab.name},
      {"root", root_node}
    });
  }

  json11::Json root = json11::Json::object{
    {"current_tab_index", layout.current_tab_index},
    {"tabs", tabs_json}
  };

  std::ofstream out(path);
  if (!out) return false;
  out << root.dump() << "\n";
  out.flush();
  return out.good();
}

std::vector<std::string> LayoutManager::availablePresets() {
  std::vector<std::string> presets;
  fs::path dir = findLayoutsDirectory();
  std::error_code ec;
  if (!fs::exists(dir, ec)) return presets;
  for (const auto &entry : fs::directory_iterator(dir, ec)) {
    if (entry.is_regular_file() && entry.path().extension() == ".json") {
      presets.push_back(entry.path().stem().string());
    }
  }
  std::sort(presets.begin(), presets.end());
  return presets;
}

fs::path LayoutManager::presetPath(const std::string &preset_name) {
  fs::path dir = findLayoutsDirectory();
  return dir / (preset_name + ".json");
}

}  // namespace cabana
