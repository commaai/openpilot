#pragma once

#include <string>

#include "tools/cabana/core/settings.h"
#include "tools/cabana/ui/util.h"

class SettingsDialog {
public:
  void open();
  void draw();

private:
  void save();

  bool open_ = false;
  PopupOwner popup_;
  int theme_ = 0;
  int cached_minutes_ = 0;
  int drag_direction_ = 0;
  int chart_height_ = 0;
  bool log_livestream_ = false;
  std::string log_path_;
};
