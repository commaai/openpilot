#include "tools/loggy/shell/runtime.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>

namespace {

constexpr const char *DEMO_ROUTE = "5beb9b58bd12b691/0000010a--a51155e496";

void print_usage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0 << " [options] [route]\n"
      << "\n"
      << "Options:\n"
      << "  --demo\n"
      << "  --preset <loggy|cabana|jotpluggler>\n"
      << "  --layout <layout>\n"
      << "  --data-dir <dir>\n"
      << "  --settings <file>\n"
      << "  --stream\n"
      << "  --address <host>\n"
      << "  --width <pixels>\n"
      << "  --height <pixels>\n"
      << "  --output <png>\n"
      << "  --show\n"
      << "  --no-hud\n";
}

bool parse_int(const char *value, int *out) {
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || end == nullptr || *end != '\0') return false;
  *out = static_cast<int>(parsed);
  return true;
}

std::string preset_from_argv0(const char *argv0) {
  const std::string name = std::filesystem::path(argv0).filename().string();
  if (name.find("cabana") != std::string::npos) return "cabana";
  if (name.find("jotpluggler") != std::string::npos) return "jotpluggler";
  return "loggy";
}

}  // namespace

int main(int argc, char *argv[]) {
  loggy::Options options;
  options.preset = preset_from_argv0(argv[0]);

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto require_value = [&](const char *flag) -> const char * {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << flag << "\n";
        print_usage(argv[0]);
        std::exit(2);
      }
      return argv[++i];
    };

    if (arg == "--demo") {
      options.route_name = DEMO_ROUTE;
    } else if (arg == "--preset") {
      options.preset = require_value("--preset");
    } else if (arg == "--layout") {
      options.layout = require_value("--layout");
    } else if (arg == "--data-dir") {
      options.data_dir = require_value("--data-dir");
    } else if (arg == "--settings") {
      options.settings_path = require_value("--settings");
    } else if (arg == "--stream") {
      options.stream = true;
    } else if (arg == "--address") {
      options.stream_address = require_value("--address");
    } else if (arg == "--output") {
      options.output_path = require_value("--output");
    } else if (arg == "--width") {
      if (!parse_int(require_value("--width"), &options.width)) {
        std::cerr << "Invalid width\n";
        return 2;
      }
    } else if (arg == "--height") {
      if (!parse_int(require_value("--height"), &options.height)) {
        std::cerr << "Invalid height\n";
        return 2;
      }
    } else if (arg == "--show") {
      options.show = true;
    } else if (arg == "--no-hud") {
      options.show_frame_hud = false;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else if (!arg.empty() && arg[0] != '-' && options.route_name.empty()) {
      options.route_name = arg;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      print_usage(argv[0]);
      return 2;
    }
  }

  if (options.output_path.empty() && !options.show) {
    options.show = true;
  }
  if (options.width <= 0 || options.height <= 0) {
    std::cerr << "Width and height must be positive\n";
    return 2;
  }
  if (options.stream && !options.route_name.empty()) {
    std::cerr << "Route/file mode and --stream are mutually exclusive\n";
    return 2;
  }

  return loggy::run(options);
}
