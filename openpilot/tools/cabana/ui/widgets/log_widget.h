#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "imgui.h"

namespace cabana {

struct CerealLogItem {
  double seconds = 0.0;
  int level = 20;
  std::string source;
  std::string message;
};

class LogWidget {
public:
  LogWidget();
  void draw();
  bool visible = false;

private:
  void updateLogs();

  uint64_t last_revision_ = UINT64_MAX;
  std::vector<CerealLogItem> logs_;
  ImGuiTextFilter filter_;
  bool show_debug_ = false;
  bool show_info_ = true;
  bool show_warning_ = true;
  bool show_error_ = true;
};

}  // namespace cabana
