#pragma once

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "tools/cabana/core/logsignals.h"

class LogPanel {
public:
  void draw();
  void loadLayout(const std::string &path);

private:
  void refresh();
  void writeFile(const std::string &path, const std::string &text);
  void fileActions();
  std::vector<cabana::LogPlot> plots_;
  std::vector<std::vector<std::vector<cabana::LogPoint>>> points_;
  std::vector<std::vector<std::vector<int>>> line_indices_;
  std::vector<std::string> names_;
  std::string filter_;
  int active_plot_ = -1;
  bool dirty_ = true;
  bool follow_ = true;
  uint64_t revision_ = std::numeric_limits<uint64_t>::max();
  double range_min_ = 0, range_max_ = 30;
  std::shared_ptr<bool> alive_ = std::make_shared<bool>(true);
};
