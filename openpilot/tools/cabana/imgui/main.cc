#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "tools/cabana/imgui/app.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/replaystream.h"

namespace {

void print_usage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0 << " [options] [route]\n"
      << "\n"
      << "the drive to replay. find your drives at connect.comma.ai\n"
      << "\n"
      << "Options:\n"
      << "  --demo                 use a demo route instead of providing your own\n"
      << "  --auto                 auto load the route from the best available source (no video)\n"
      << "  --qcam                 load qcamera\n"
      << "  --ecam                 load wide road camera\n"
      << "  --dcam                 load driver camera\n"
      << "  --msgq                 read can messages from the msgq\n"
      << "  --panda                read can messages from panda\n"
      << "  --panda-serial <s>     read can messages from panda with given serial\n"
      << "  --socketcan <dev>      read can messages from given SocketCAN device\n"
      << "  --zmq <ip-address>     read can messages from zmq at the specified ip-address\n"
      << "  --data_dir <dir>       local directory with routes\n"
      << "  --no-vipc              do not output video\n"
      << "  --dbc <file>           dbc file to open\n"
      << "  --width <pixels>       window width (default 1600)\n"
      << "  --height <pixels>      window height (default 900)\n"
      << "  --output <png>         render headless and capture a screenshot\n"
      << "  --theme <light|dark>   color theme (default light)\n";
}

bool parse_int(const char *value, int *out) {
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == nullptr || *end != '\0') return false;
  *out = static_cast<int>(parsed);
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

    if (arg == "--demo") {
      options.route = DEMO_ROUTE;
    } else if (arg == "--auto") {
      options.auto_source = true;
    } else if (arg == "--qcam") {
      options.qcam = true;
    } else if (arg == "--ecam") {
      options.ecam = true;
    } else if (arg == "--dcam") {
      options.dcam = true;
    } else if (arg == "--msgq") {
      options.msgq = true;
    } else if (arg == "--panda") {
      options.panda = true;
    } else if (arg == "--panda-serial") {
      options.panda_serial = require_value("--panda-serial");
    } else if (arg == "--socketcan") {
      options.socketcan_device = require_value("--socketcan");
    } else if (arg == "--zmq") {
      options.zmq_address = require_value("--zmq");
    } else if (arg == "--data_dir") {
      options.data_dir = require_value("--data_dir");
    } else if (arg == "--no-vipc") {
      options.no_vipc = true;
    } else if (arg == "--dbc") {
      options.dbc_path = require_value("--dbc");
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
    } else if (arg == "--output") {
      options.output_path = require_value("--output");
      options.show = false;
    } else if (arg == "--theme") {
      const std::string theme = require_value("--theme");
      if (theme != "light" && theme != "dark") {
        std::cerr << "Invalid theme: " << theme << "\n";
        return 2;
      }
      options.dark_theme = (theme == "dark");
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else if (!arg.empty() && arg[0] != '-' && options.route.empty()) {
      options.route = arg;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      print_usage(argv[0]);
      return 2;
    }
  }

  if (options.width <= 0 || options.height <= 0) {
    std::cerr << "Width and height must be positive\n";
    return 2;
  }
  if (options.msgq || options.panda || !options.panda_serial.empty() ||
      !options.socketcan_device.empty() || !options.zmq_address.empty()) {
    std::cerr << "cabana(imgui): live streams are not wired up yet (Phase 6 of MIGRATION.md); starting empty.\n";
  }
  if (!options.dbc_path.empty()) {
    std::string error;
    if (!dbc()->open(SOURCE_ALL, options.dbc_path, &error)) {
      std::cerr << "Failed to load DBC file " << options.dbc_path << ": " << error << "\n";
      return 1;
    }
  }

  std::unique_ptr<AbstractStream> stream;
  if (!options.route.empty()) {
    // flag mapping mirrors the old Qt tools/cabana/cabana.cc main()
    uint32_t replay_flags = REPLAY_FLAG_NONE;
    if (options.ecam) replay_flags |= REPLAY_FLAG_ECAM;
    if (options.qcam) replay_flags |= REPLAY_FLAG_QCAMERA;
    if (options.dcam) replay_flags |= REPLAY_FLAG_DCAM;
    if (options.no_vipc) replay_flags |= REPLAY_FLAG_NO_VIPC;

    auto replay_stream = std::make_unique<ReplayStream>();
    if (!replay_stream->loadRoute(options.route, options.data_dir, replay_flags, options.auto_source)) {
      return 1;  // error already printed by loadRoute
    }
    stream = std::move(replay_stream);
  } else {
    stream = std::make_unique<DummyStream>();
  }

  return run(options, std::move(stream));
}
