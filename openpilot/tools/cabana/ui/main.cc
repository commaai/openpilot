#include <cstdio>
#include <cstring>
#include <string>

#include "tools/cabana/ui/app.h"

namespace {

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

// Returns 0 to continue, or a process exit code (0 for --help, 1 for errors).
int parseArgs(int argc, char *argv[], Options &opts, bool &ok) {
  ok = false;
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (std::strcmp(a, "--help") == 0 || std::strcmp(a, "-h") == 0) {
      printUsage(argv[0]);
      return 0;
    } else if (std::strcmp(a, "--demo") == 0) {
      opts.demo = true;
    } else if (std::strcmp(a, "--auto") == 0) {
      opts.auto_source = true;
    } else if (std::strcmp(a, "--qcam") == 0) {
      opts.qcam = true;
    } else if (std::strcmp(a, "--wide-road") == 0 || std::strcmp(a, "--ecam") == 0) {
      opts.wide_road = true;
    } else if (std::strcmp(a, "--cabin") == 0 || std::strcmp(a, "--dcam") == 0) {
      opts.cabin = true;
    } else if (std::strcmp(a, "--msgq") == 0) {
      opts.msgq = true;
    } else if (std::strcmp(a, "--panda") == 0) {
      opts.panda = true;
    } else if (std::strcmp(a, "--panda-serial") == 0) {
      if (!takeValue(argc, argv, i, opts.panda_serial)) return 1;
      opts.panda = true;
    } else if (std::strcmp(a, "--socketcan") == 0) {
      if (!takeValue(argc, argv, i, opts.socketcan)) return 1;
#ifndef __linux__
      fprintf(stderr, "error: --socketcan is only supported on Linux\n");
      return 1;
#endif
    } else if (std::strcmp(a, "--zmq") == 0) {
      if (!takeValue(argc, argv, i, opts.zmq)) return 1;
    } else if (std::strcmp(a, "--data_dir") == 0) {
      if (!takeValue(argc, argv, i, opts.data_dir)) return 1;
    } else if (std::strcmp(a, "--no-vipc") == 0) {
      opts.no_vipc = true;
    } else if (std::strcmp(a, "--dbc") == 0) {
      if (!takeValue(argc, argv, i, opts.dbc)) return 1;
    } else if (a[0] == '-') {
      fprintf(stderr, "error: unknown option %s\n", a);
      printUsage(argv[0]);
      return 1;
    } else if (opts.route.empty()) {
      opts.route = a;
    } else {
      fprintf(stderr, "error: unexpected argument %s\n", a);
      printUsage(argv[0]);
      return 1;
    }
  }
  ok = true;
  return 0;
}

}  // namespace

int main(int argc, char *argv[]) {
  Options opts;
  bool ok = false;
  if (const int code = parseArgs(argc, argv, opts, ok); !ok) {
    return code;
  }
  return run(opts);
}
