#include <getopt.h>

#include <QApplication>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "common/prefix.h"
#include "tools/replay/consoleui.h"
#include "tools/replay/replay.h"
#include "tools/replay/util.h"

const std::string helpText =
R"(Usage: replay [options]
Options:
  -a, --allow        Whitelist of services to send
  -b, --block        Blacklist of services to send
  -c, --cache        Cache <n> segments in memory. Default is 5
  -s, --start        Start from <seconds>
  -x, --playback     Playback <speed>
      --demo         Use a demo route instead of providing your own
  -d, --data_dir     Local directory with routes
  -p, --prefix       Set OPENPILOT_PREFIX
      --dcam         Load driver camera
      --ecam         Load wide road camera
      --no-loop      Stop at the end of the route
      --no-cache     Turn off local cache
      --qcam         Load qcamera
      --no-hw-decoder Disable HW video decoding
      --no-vipc      Do not output video
      --all          Output all messages including uiDebug, userFlag
  -h, --help         Show this help message
)";

struct ReplayConfig {
  std::string route;
  std::vector<std::string> allow;
  std::vector<std::string> block;
  std::string data_dir;
  std::string prefix;
  uint32_t flags = REPLAY_FLAG_NONE;
  int start_seconds = 0;
  int cache_segments = -1;
  float playback_speed = -1;
};

bool parseArgs(int argc, char *argv[], ReplayConfig &config) {
  const struct option cli_options[] = {
      {"allow", required_argument, nullptr, 'a'},
      {"block", required_argument, nullptr, 'b'},
      {"cache", required_argument, nullptr, 'c'},
      {"start", required_argument, nullptr, 's'},
      {"playback", required_argument, nullptr, 'x'},
      {"demo", no_argument, nullptr, 0},
      {"data_dir", required_argument, nullptr, 'd'},
      {"prefix", required_argument, nullptr, 'p'},
      {"dcam", no_argument, nullptr, 0},
      {"ecam", no_argument, nullptr, 0},
      {"no-loop", no_argument, nullptr, 0},
      {"no-cache", no_argument, nullptr, 0},
      {"qcam", no_argument, nullptr, 0},
      {"no-hw-decoder", no_argument, nullptr, 0},
      {"no-vipc", no_argument, nullptr, 0},
      {"all", no_argument, nullptr, 0},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0},  // Terminating entry
  };

  const std::map<std::string, REPLAY_FLAGS> flag_map = {
      {"dcam", REPLAY_FLAG_DCAM},
      {"ecam", REPLAY_FLAG_ECAM},
      {"no-loop", REPLAY_FLAG_NO_LOOP},
      {"no-cache", REPLAY_FLAG_NO_FILE_CACHE},
      {"qcam", REPLAY_FLAG_QCAMERA},
      {"no-hw-decoder", REPLAY_FLAG_NO_HW_DECODER},
      {"no-vipc", REPLAY_FLAG_NO_VIPC},
      {"all", REPLAY_FLAG_ALL_SERVICES},
  };

  if (argc == 1) {
    std::cout << helpText;
    return false;
  }

  int opt, option_index = 0;
  while ((opt = getopt_long(argc, argv, "a:b:c:s:x:d:p:h", cli_options, &option_index)) != -1) {
    switch (opt) {
      case 'a': config.allow = split(optarg, ','); break;
      case 'b': config.block = split(optarg, ','); break;
      case 'c': config.cache_segments = std::atoi(optarg); break;
      case 's': config.start_seconds = std::atoi(optarg); break;
      case 'x': config.playback_speed = std::atof(optarg); break;
      case 'd': config.data_dir = optarg; break;
      case 'p': config.prefix = optarg; break;
      case 0: {
        std::string name = cli_options[option_index].name;
        if (name == "demo") {
          config.route = DEMO_ROUTE;
        } else {
          config.flags |= flag_map.at(name);
        }
        break;
      }
      case 'h': std::cout << helpText; return false;
      default: return false;
    }
  }

  // Check for a route name (first positional argument)
  if (config.route.empty() && optind < argc) {
    config.route = argv[optind];
  }

  if (config.route.empty()) {
    std::cerr << "No route provided. Use --help for usage information.\n";
    return false;
  }

  return true;
}

int main(int argc, char *argv[]) {
#ifdef __APPLE__
  // With all sockets opened, we might hit the default limit of 256 on macOS
  util::set_file_descriptor_limit(1024);
#endif

  QCoreApplication app(argc, argv);
  ReplayConfig config;

  if (!parseArgs(argc, argv, config)) {
    return 1;
  }

  std::unique_ptr<OpenpilotPrefix> op_prefix;
  if (!config.prefix.empty()) {
    op_prefix = std::make_unique<OpenpilotPrefix>(config.prefix);
  }

  Replay *replay = new Replay(config.route, config.allow, config.block, nullptr, config.flags, config.data_dir, &app);
  if (config.cache_segments > 0) {
    replay->setSegmentCacheLimit(config.cache_segments);
  }
  if (config.playback_speed > 0) {
    replay->setSpeed(std::clamp(config.playback_speed, ConsoleUI::speed_array.front(), ConsoleUI::speed_array.back()));
  }
  if (!replay->load()) {
    return 1;
  }

  ConsoleUI console_ui(replay);
  replay->start(config.start_seconds);
  return app.exec();
}
