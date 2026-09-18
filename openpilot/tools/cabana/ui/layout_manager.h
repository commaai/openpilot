#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tools/cabana/ui/custom_eval.h"

namespace cabana {

struct LayoutCurve {
  std::string name;
  std::string color_hex;
  std::string can_id;  // empty for cereal/custom curves
  bool visible = true;
  std::optional<CustomPythonSeries> custom_python;
  bool derivative = false;
  double derivative_dt = 0.0;
  double scale = 1.0;
  double offset = 0.0;
};

struct LayoutPane {
  std::string title;
  int series_type = 0;
  std::string kind;  // "map", "logs", or ""
  std::optional<std::pair<double, double>> y_limits;
  std::vector<LayoutCurve> curves;
};

struct LayoutTab {
  std::string name;
  std::vector<LayoutPane> panes;
};

struct Layout {
  std::string name;
  int current_tab_index = 0;
  std::vector<LayoutTab> tabs;
};

class LayoutManager {
public:
  static Layout loadLayout(const std::filesystem::path &path);
  static bool saveLayout(const Layout &layout, const std::filesystem::path &path);
  static std::vector<std::string> availablePresets();
  static std::filesystem::path presetPath(const std::string &preset_name);
};

}  // namespace cabana
