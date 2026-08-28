#pragma once

#include <array>
#include <string>
#include <vector>

#include "tools/cabana/ui/tools/tooldialog.h"

class RouteInfoDlg : public ToolDialog {
public:
  RouteInfoDlg();
  bool draw() override;

private:
  std::string title_;
  std::vector<std::array<std::string, 7>> rows_;  // one row per segment: seg num, rlog, narrow road, wide road, driver, qlog, qcam
  bool open_ = true;
};
