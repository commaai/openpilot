#include <cstdlib>
#include <iostream>

#include "tools/jotpluggler/app.h"

namespace {

constexpr const char *DEMO_ROUTE = "5beb9b58bd12b691/0000010a--a51155e496";

void print_usage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0 << " [--layout <layout>] [options] [route]\n"
      << "\n"
      << "Options:\n"
      << "  --demo\n"
      << "  --data-dir <dir>\n"
      << "  --stream\n"
      << "  --address <host>\n"
      << "  --buffer-seconds <seconds>\n"
      << "  --width <pixels>\n"
      << "  --height <pixels>\n"
      << "  --output <png>\n"
      << "  --show\n"
      << "  --sync-load\n"
      << "\n"
      << "Examples:\n"
      << "  " << argv0 << "\n"
      << "  " << argv0 << " --demo\n"
      << "  " << argv0 << " --layout longitudinal --demo\n"
      << "  " << argv0 << " --layout longitudinal --demo --output /tmp/longitudinal.png\n"
      << "  " << argv0 << " --stream --show\n"
      << "  " << argv0 << " --stream --address 192.168.60.52 --buffer-seconds 45 --show\n";
}

bool parse_int(const char *value, int *out) {
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == nullptr || *end != '\0') return false;
  *out = static_cast<int>(parsed);
  return true;
}

bool parse_double(const char *value, double *out) {
  char *end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == nullptr || *end != '\0') return false;
  *out = parsed;
  return true;
}

}  // namespace

int main(int argc, char *argv[]) {
  Options options;
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

    if (arg == "--layout") {
      options.layout = require_value("--layout");
    } else if (arg == "--demo") {
      options.route_name = DEMO_ROUTE;
    } else if (arg == "--data-dir") {
      options.data_dir = require_value("--data-dir");
    } else if (arg == "--stream") {
      options.stream = true;
    } else if (arg == "--address") {
      options.stream_address = require_value("--address");
    } else if (arg == "--buffer-seconds") {
      if (!parse_double(require_value("--buffer-seconds"), &options.stream_buffer_seconds)) {
        std::cerr << "Invalid buffer seconds\n";
        return 2;
      }
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
    } else if (arg == "--sync-load") {
      options.sync_load = true;
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
  if (options.stream_buffer_seconds <= 0.0) {
    std::cerr << "Buffer seconds must be positive\n";
    return 2;
  }

  return run(options);
}
