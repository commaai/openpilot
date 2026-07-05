#pragma once

#include <string>

namespace loggy {

struct Options {
  std::string preset = "loggy";
  std::string layout;
  std::string route_name;
  std::string data_dir;
  std::string settings_path;
  std::string output_path;
  std::string stream_address = "127.0.0.1";
  int width = 1600;
  int height = 900;
  bool show = false;
  bool stream = false;
  bool show_frame_hud = true;
};

int run(const Options &options);

}  // namespace loggy
