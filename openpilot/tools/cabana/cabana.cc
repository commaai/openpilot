#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include <QApplication>

#include "tools/cabana/mainwin.h"
#include "tools/cabana/streams/devicestream.h"
#include "tools/cabana/streams/pandastream.h"
#include "tools/cabana/streams/replaystream.h"
#ifdef __linux__
#include "tools/cabana/streams/socketcanstream.h"
#endif

namespace {

struct CabanaArgs {
  bool demo = false;
  bool auto_source = false;
  bool qcam = false;
  bool ecam = false;
  bool dcam = false;
  bool msgq = false;
  bool panda = false;
  bool no_vipc = false;
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
          "  --ecam                    load wide road camera\n"
          "  --dcam                    load driver camera\n"
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

// Returns true if value was consumed from argv[i+1].
bool takeValue(int argc, char *argv[], int &i, std::string &out) {
  if (i + 1 >= argc) {
    fprintf(stderr, "error: %s requires a value\n", argv[i]);
    return false;
  }
  out = argv[++i];
  return true;
}

// Returns 0 to continue, or a process exit code (0 for --help, 1 for errors).
int parseArgs(int argc, char *argv[], CabanaArgs &args, bool &ok) {
  ok = false;
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
    } else if (std::strcmp(a, "--ecam") == 0) {
      args.ecam = true;
    } else if (std::strcmp(a, "--dcam") == 0) {
      args.dcam = true;
    } else if (std::strcmp(a, "--msgq") == 0) {
      args.msgq = true;
    } else if (std::strcmp(a, "--panda") == 0) {
      args.panda = true;
    } else if (std::strcmp(a, "--panda-serial") == 0) {
      if (!takeValue(argc, argv, i, args.panda_serial)) return 1;
      args.panda = true;
    } else if (std::strcmp(a, "--socketcan") == 0) {
      if (!takeValue(argc, argv, i, args.socketcan)) return 1;
#ifdef __linux__
#else
      fprintf(stderr, "error: --socketcan is only supported on Linux\n");
      return 1;
#endif
    } else if (std::strcmp(a, "--zmq") == 0) {
      if (!takeValue(argc, argv, i, args.zmq)) return 1;
    } else if (std::strcmp(a, "--data_dir") == 0) {
      if (!takeValue(argc, argv, i, args.data_dir)) return 1;
    } else if (std::strcmp(a, "--no-vipc") == 0) {
      args.no_vipc = true;
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
  ok = true;
  return 0;
}

}  // namespace

int main(int argc, char *argv[]) {
  QCoreApplication::setApplicationName("Cabana");
  initApp(argc, argv, false);
  QApplication app(argc, argv);
  app.setApplicationDisplayName("Cabana");
  //app.setWindowIcon(QIcon(":cabana-icon.png"));  // TODO: do this in imgui

  UnixSignalHandler signalHandler;
  utils::setTheme(settings.theme);

  CabanaArgs args;
  bool args_ok = false;
  if (const int code = parseArgs(argc, argv, args, args_ok); !args_ok) {
    return code;
  }

  AbstractStream *stream = nullptr;

  if (args.msgq) {
    stream = new DeviceStream(&app);
  } else if (!args.zmq.empty()) {
    stream = new DeviceStream(&app, QString::fromStdString(args.zmq));
  } else if (args.panda || !args.panda_serial.empty()) {
    try {
      stream = new PandaStream(&app, {.serial = args.panda_serial});
    } catch (std::exception &e) {
      fprintf(stderr, "%s\n", e.what());
      return 0;
    }
#ifdef __linux__
  } else if (SocketCanStream::available() && !args.socketcan.empty()) {
    stream = new SocketCanStream(&app, {.device = args.socketcan});
#endif
  } else {
    uint32_t replay_flags = REPLAY_FLAG_NONE;
    if (args.ecam) replay_flags |= REPLAY_FLAG_ECAM;
    if (args.qcam) replay_flags |= REPLAY_FLAG_QCAMERA;
    if (args.dcam) replay_flags |= REPLAY_FLAG_DCAM;
    if (args.no_vipc) replay_flags |= REPLAY_FLAG_NO_VIPC;

    QString route;
    if (!args.route.empty()) {
      route = QString::fromStdString(args.route);
    } else if (args.demo) {
      route = DEMO_ROUTE;
    }
    if (!route.isEmpty()) {
      auto replay_stream = std::make_unique<ReplayStream>(&app);
      if (!replay_stream->loadRoute(route.toStdString(), args.data_dir, replay_flags, args.auto_source)) {
        return 0;
      }
      stream = replay_stream.release();
    }
  }

  MainWindow w(stream, QString::fromStdString(args.dbc));
  return app.exec();
}
