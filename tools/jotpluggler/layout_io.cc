#include "tools/jotpluggler/app.h"
#include "tools/jotpluggler/common.h"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "third_party/json11/json11.hpp"

namespace fs = std::filesystem;

namespace {

std::string curve_color_hex(const std::array<uint8_t, 3> &color) {
  std::ostringstream hex;
  hex << "#" << std::hex << std::setfill('0')
      << std::setw(2) << static_cast<int>(color[0])
      << std::setw(2) << static_cast<int>(color[1])
      << std::setw(2) << static_cast<int>(color[2]);
  return hex.str();
}

json11::Json curve_to_json(const Curve &curve) {
  json11::Json::object obj = {
    {"name", curve.name},
    {"color", curve_color_hex(curve.color)},
  };
  if (curve.derivative) {
    obj["transform"] = "derivative";
    if (curve.derivative_dt > 0.0) {
      obj["derivative_dt"] = curve.derivative_dt;
    }
  } else if (std::abs(curve.value_scale - 1.0) > 1.0e-9 || std::abs(curve.value_offset) > 1.0e-9) {
    obj["transform"] = "scale";
    obj["scale"] = curve.value_scale;
    obj["offset"] = curve.value_offset;
  }
  if (curve.custom_python.has_value()) {
    json11::Json::array additional_sources;
    for (const std::string &path : curve.custom_python->additional_sources) {
      additional_sources.push_back(path);
    }
    obj["custom_python"] = json11::Json::object{
      {"linked_source", curve.custom_python->linked_source},
      {"additional_sources", additional_sources},
      {"globals_code", curve.custom_python->globals_code},
      {"function_code", curve.custom_python->function_code},
    };
  }
  return obj;
}

json11::Json workspace_node_to_json(const WorkspaceNode &node, const WorkspaceTab &tab) {
  if (node.is_pane) {
    if (node.pane_index < 0 || node.pane_index >= static_cast<int>(tab.panes.size())) {
      return nullptr;
    }
    const Pane &pane = tab.panes[static_cast<size_t>(node.pane_index)];
    json11::Json::object obj = {
      {"title", pane.title.empty() ? std::string("...") : pane.title},
    };
    if (pane.kind == PaneKind::Map) {
      obj["kind"] = "map";
    } else if (pane.kind == PaneKind::Camera) {
      obj["kind"] = "camera";
      obj["camera_view"] = camera_view_spec(pane.camera_view).layout_name;
    }
    if (pane.range.valid) {
      obj["range"] = json11::Json::object{
        {"left", pane.range.left}, {"right", pane.range.right},
        {"top", pane.range.top}, {"bottom", pane.range.bottom},
      };
    }
    if (pane.range.has_y_limit_min || pane.range.has_y_limit_max) {
      json11::Json::object limits;
      if (pane.range.has_y_limit_min) {
        limits["min"] = pane.range.y_limit_min;
      }
      if (pane.range.has_y_limit_max) {
        limits["max"] = pane.range.y_limit_max;
      }
      obj["y_limits"] = limits;
    }
    json11::Json::array curves;
    for (const Curve &curve : pane.curves) {
      if (!curve.runtime_only) {
        curves.push_back(curve_to_json(curve));
      }
    }
    obj["curves"] = curves;
    return obj;
  }

  if (node.children.empty()) return nullptr;
  json11::Json::array sizes;
  for (size_t i = 0; i < node.children.size(); ++i) {
    sizes.push_back(i < node.sizes.size() ? static_cast<double>(node.sizes[i])
                                          : 1.0 / static_cast<double>(node.children.size()));
  }
  json11::Json::array children;
  for (const WorkspaceNode &child : node.children) {
    children.push_back(workspace_node_to_json(child, tab));
  }
  return json11::Json::object{
    {"split", node.orientation == SplitOrientation::Horizontal ? "horizontal" : "vertical"},
    {"sizes", sizes},
    {"children", children},
  };
}

}  // namespace

void save_layout_json(const SketchLayout &layout, const fs::path &path) {
  ensure_parent_dir(path);
  json11::Json::array tabs;
  for (const WorkspaceTab &tab : layout.tabs) {
    tabs.push_back(json11::Json::object{
      {"name", tab.tab_name},
      {"root", workspace_node_to_json(tab.root, tab)},
    });
  }
  const json11::Json root = json11::Json::object{
    {"current_tab_index", std::clamp(layout.current_tab_index, 0, std::max(0, static_cast<int>(layout.tabs.size()) - 1))},
    {"tabs", tabs},
  };
  write_file_or_throw(path, root.dump() + "\n");
}
