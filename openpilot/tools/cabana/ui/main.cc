#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "tools/cabana/streams/devicestream.h"
#include "tools/cabana/streams/pandastream.h"
#include "tools/cabana/streams/replaystream.h"
#ifdef __linux__
#include "tools/cabana/streams/socketcanstream.h"
#endif
#include "tools/cabana/ui/app.h"
#include "tools/cabana/ui/dialogs/messagebox.h"
#include "tools/cabana/utils/util.h"

#ifdef __GLIBC__
#include <malloc.h>
#endif

namespace {

struct CabanaArgs {
  bool demo = false;
  bool auto_source = false;
  bool qcam = false;
  bool wide_road = false;
  bool cabin = false;
  bool msgq = false;
  bool panda = false;
  bool no_vipc = false;
  bool no_cache = false;
  std::string panda_serial;
  std::string socketcan;
  std::string zmq;
  std::string data_dir;
  std::string dbc;
  std::string route;
};

void printUsage(const char *argv0) {
  fprintf(stderr,
          "Usage: %s [options] [route]\n"
          "\n"
          "  route                     the drive to replay. find your drives at connect.comma.ai\n"
          "\n"
          "Options:\n"
          "  --help                    show this help\n"
          "  --demo                    use a demo route instead of providing your own\n"
          "  --auto                    Auto load the route from the best available source (no video):\n"
          "                            internal, openpilotci, comma_api, car_segments, testing_closet\n"
          "  --qcam                    load qcamera\n"
          "  --wide-road               load wide road camera (alias: --ecam)\n"
          "  --cabin                   load cabin camera (alias: --dcam)\n"
          "  --msgq                    read can messages from the msgq\n"
          "  --panda                   read can messages from panda\n"
          "  --panda-serial <serial>   read can messages from panda with given serial\n"
#ifdef __linux__
          "  --socketcan <device>      read can messages from given SocketCAN device\n"
#endif
          "  --zmq <ip-address>        read can messages from zmq at the specified ip-address\n"
          "  --data_dir <dir>          local directory with routes\n"
          "  --no-vipc                 do not output video\n"
          "  --no-cache                turn off the local route file cache\n"
          "  --dbc <file>              dbc file to open\n",
          argv0);
}

bool takeValue(int argc, char *argv[], int &i, std::string &out) {
  if (i + 1 >= argc) {
    fprintf(stderr, "error: %s requires a value\n", argv[i]);
    return false;
  }
  out = argv[++i];
  return true;
}

// the process exit code, or nullopt to continue
std::optional<int> parseArgs(int argc, char *argv[], CabanaArgs &args) {
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (std::strcmp(a, "--help") == 0 || std::strcmp(a, "-h") == 0) {
      printUsage(argv[0]);
      return 0;
    } else if (std::strcmp(a, "--demo") == 0) {
      args.demo = true;
    } else if (std::strcmp(a, "--auto") == 0) {
      args.auto_source = true;
    } else if (std::strcmp(a, "--qcam") == 0) {
      args.qcam = true;
    } else if (std::strcmp(a, "--wide-road") == 0 || std::strcmp(a, "--ecam") == 0) {
      args.wide_road = true;
    } else if (std::strcmp(a, "--cabin") == 0 || std::strcmp(a, "--dcam") == 0) {
      args.cabin = true;
    } else if (std::strcmp(a, "--msgq") == 0) {
      args.msgq = true;
    } else if (std::strcmp(a, "--panda") == 0) {
      args.panda = true;
    } else if (std::strcmp(a, "--panda-serial") == 0) {
      if (!takeValue(argc, argv, i, args.panda_serial)) return 1;
      args.panda = true;
    } else if (std::strcmp(a, "--socketcan") == 0) {
      if (!takeValue(argc, argv, i, args.socketcan)) return 1;
#ifndef __linux__
      fprintf(stderr, "error: --socketcan is only supported on Linux\n");
      return 1;
#endif
    } else if (std::strcmp(a, "--zmq") == 0) {
      if (!takeValue(argc, argv, i, args.zmq)) return 1;
    } else if (std::strcmp(a, "--data_dir") == 0) {
      if (!takeValue(argc, argv, i, args.data_dir)) return 1;
    } else if (std::strcmp(a, "--no-vipc") == 0) {
      args.no_vipc = true;
    } else if (std::strcmp(a, "--no-cache") == 0) {
      args.no_cache = true;
    } else if (std::strcmp(a, "--dbc") == 0) {
      if (!takeValue(argc, argv, i, args.dbc)) return 1;
    } else if (a[0] == '-') {
      fprintf(stderr, "error: unknown option %s\n", a);
      printUsage(argv[0]);
      return 1;
    } else if (args.route.empty()) {
      args.route = a;
    } else {
      fprintf(stderr, "error: unexpected argument %s\n", a);
      printUsage(argv[0]);
      return 1;
    }
  }
  return std::nullopt;
}

}  // namespace

int main(int argc, char *argv[]) {
#ifdef __GLIBC__
  // Worker threads (sparklines, chart series, replay) would each get their own glibc malloc arena and the
  // arenas fragment without bound (RSS grew ~3 MB/min with charts open). macOS has a single allocator zone.
  mallopt(M_ARENA_MAX, 1);
#endif
  // ensure the current dir matches the executable's directory
  std::error_code ec;
  std::filesystem::current_path(executableDir(), ec);

  CabanaArgs args;
  if (auto code = parseArgs(argc, argv, args)) return *code;

  std::unique_ptr<AbstractStream> stream;
  StreamLoader stream_loader;

  if (args.msgq) {
    stream = std::make_unique<DeviceStream>();
  } else if (!args.zmq.empty()) {
    stream = std::make_unique<DeviceStream>(args.zmq);
  } else if (args.panda) {
    try {
      stream = std::make_unique<PandaStream>(PandaStreamConfig{.serial = args.panda_serial});
    } catch (std::exception &e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
    }
#ifdef __linux__
  } else if (!args.socketcan.empty()) {
    if (!SocketCanStream::available()) {
      fprintf(stderr, "error: SocketCAN is not available on this system\n");
      return 1;
    }
    stream = std::make_unique<SocketCanStream>(SocketCanStreamConfig{.device = args.socketcan});
#endif
  } else {
    uint32_t replay_flags = REPLAY_FLAG_NONE;
    if (args.wide_road) replay_flags |= REPLAY_FLAG_WIDE_ROAD;
    if (args.qcam) replay_flags |= REPLAY_FLAG_QCAMERA;
    if (args.cabin) replay_flags |= REPLAY_FLAG_CABIN_CAMERA;
    if (args.no_vipc) replay_flags |= REPLAY_FLAG_NO_VIPC;
    if (args.no_cache) replay_flags |= REPLAY_FLAG_NO_FILE_CACHE;

    std::string route;
    if (!args.route.empty()) {
      route = args.route;
    } else if (args.demo) {
      route = DEMO_ROUTE;
    }
    if (!route.empty()) {
      // the route file listing hits the comma API; load behind the window instead of before it
      stream_loader = [route, data_dir = args.data_dir, replay_flags, auto_source = args.auto_source]() -> std::unique_ptr<AbstractStream> {
        auto replay_stream = std::make_unique<ReplayStream>();
        Connection err = replay_stream->error.connect([](const std::string &msg) {
          fprintf(stderr, "%s\n", msg.c_str());
          utils::runOnMainThread([msg]() { MessageBox::warning("Error", msg); });
        });
        if (!replay_stream->loadRoute(route, data_dir, replay_flags, auto_source)) {
          return nullptr;
        }
        return replay_stream;
      };
    }
  }

  return run(std::move(stream), std::move(stream_loader), args.dbc);
}
