#include "tools/loggy/shell/runtime.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

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
      << "  --device <host>       start_ cereal bridge for a comma device\n"
      << "  --panda               read CAN from first Panda USB\n"
      << "  --panda-serial <serial>\n"
      << "  --panda-bus <bus>:<can_kbps>[:fd|off[:data_kbps]]\n"
      << "  --socketcan <dev>\n"
      << "  --stream-buffer <seconds>\n"
      << "  --width <pixels>\n"
      << "  --height <pixels>\n"
      << "  --output <png>\n"
      << "  --show\n"
      << "  --no-hud\n";
}

bool parse_int(const char *value, int &out) {
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || end == nullptr || *end != '\0') return false;
  out = static_cast<int>(parsed);
  return true;
}

bool parse_double(const char *value, double &out) {
  char *end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value || end == nullptr || *end != '\0') return false;
  out = parsed;
  return true;
}

std::vector<std::string> split_token(std::string value, char delimiter) {
  std::vector<std::string> parts;
  size_t start_ = 0;
  while (start_ <= value.size()) {
    const size_t end = value.find(delimiter, start_);
    parts.push_back(value.substr(start_, end == std::string::npos ? std::string::npos : end - start_));
    if (end == std::string::npos) break;
    start_ = end + 1;
  }
  return parts;
}

bool parse_uint16_token(const std::string &value, uint16_t &out) {
  if (value.empty()) return false;
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value.c_str(), &end, 10);
  if (end == value.c_str() || end == nullptr || *end != '\0' || parsed > UINT16_MAX) return false;
  out = static_cast<uint16_t>(parsed);
  return true;
}

bool parse_bool_token(const std::string &value, bool &out) {
  if (value == "1" || value == "true" || value == "on" || value == "fd") {
    out = true;
    return true;
  }
  if (value == "0" || value == "false" || value == "off" || value == "nofd") {
    out = false;
    return true;
  }
  return false;
}

bool parse_panda_bus_arg(const std::string &value,
                         std::array<loggy::PandaBusConfig, loggy::kPandaBusCount> &buses,
                         std::string &error) {
  const std::vector<std::string> parts = split_token(value, ':');
  if (parts.size() < 2 || parts.size() > 4) {
    error = "expected bus:can_kbps[:fd|off[:data_kbps]]";
    return false;
  }
  uint16_t bus = 0;
  uint16_t can_speed = 0;
  if (!parse_uint16_token(parts[0], bus) || bus >= loggy::kPandaBusCount) {
    error = "invalid Panda bus index";
    return false;
  }
  if (!parse_uint16_token(parts[1], can_speed) || !loggy::live_panda_can_speed_supported(can_speed)) {
    error = "invalid Panda CAN speed";
    return false;
  }
  loggy::PandaBusConfig config = buses[bus];
  config.can_speed_kbps = can_speed;
  if (parts.size() >= 3 && !parse_bool_token(parts[2], config.can_fd)) {
    error = "invalid Panda FD flag";
    return false;
  }
  if (parts.size() >= 4 &&
      (!parse_uint16_token(parts[3], config.data_speed_kbps) ||
       !loggy::live_panda_data_speed_supported(config.data_speed_kbps))) {
    error = "invalid Panda data speed";
    return false;
  }
  buses[bus] = loggy::normalize_live_panda_bus_config(config);
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
      options.stream_source_kind = loggy::live_is_local_stream_address(options.stream_address)
        ? loggy::LiveSourceKind::CerealLocal
        : loggy::LiveSourceKind::CerealRemote;
    } else if (arg == "--device") {
      options.stream = true;
      options.stream_source_kind = loggy::LiveSourceKind::DeviceBridge;
      options.stream_address = require_value("--device");
    } else if (arg == "--panda") {
      options.stream = true;
      options.stream_source_kind = loggy::LiveSourceKind::PandaUsb;
      options.stream_address.clear();
    } else if (arg == "--panda-serial") {
      options.stream = true;
      options.stream_source_kind = loggy::LiveSourceKind::PandaUsb;
      options.stream_address = require_value("--panda-serial");
    } else if (arg == "--panda-bus") {
      if (options.stream_source_kind != loggy::LiveSourceKind::PandaUsb) {
        options.stream_address.clear();
      }
      options.stream = true;
      options.stream_source_kind = loggy::LiveSourceKind::PandaUsb;
      std::string error;
      if (!parse_panda_bus_arg(require_value("--panda-bus"), options.stream_panda_buses, error)) {
        std::cerr << "Invalid Panda bus config: " << error << "\n";
        return 2;
      }
    } else if (arg == "--socketcan") {
      options.stream = true;
      options.stream_source_kind = loggy::LiveSourceKind::SocketCan;
      options.stream_address = require_value("--socketcan");
    } else if (arg == "--stream-buffer") {
      if (!parse_double(require_value("--stream-buffer"), options.stream_buffer_seconds) ||
          options.stream_buffer_seconds < 1.0) {
        std::cerr << "Invalid stream buffer\n";
        return 2;
      }
    } else if (arg == "--output") {
      options.output_path = require_value("--output");
    } else if (arg == "--width") {
      if (!parse_int(require_value("--width"), options.width)) {
        std::cerr << "Invalid width\n";
        return 2;
      }
    } else if (arg == "--height") {
      if (!parse_int(require_value("--height"), options.height)) {
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
