#include <QApplication>
#include <QCommandLineParser>

#include "selfdrive/ui/qt/util.h"
#include "tools/cabana/mainwin.h"
#include "tools/cabana/streams/devicestream.h"
#include "tools/cabana/streams/pandastream.h"
#include "tools/cabana/streams/replaystream.h"
#include "tools/cabana/streams/socketcanstream.h"

int main(int argc, char *argv[]) {
  QCoreApplication::setApplicationName("Cabana");
  QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
  initApp(argc, argv, false);
  QApplication app(argc, argv);
  app.setApplicationDisplayName("Cabana");
  app.setWindowIcon(QIcon(":cabana-icon.png"));

  UnixSignalHandler signalHandler;
  utils::setTheme(settings.theme);

  QCommandLineParser cmd_parser;
  cmd_parser.addHelpOption();
  cmd_parser.addPositionalArgument("route", "the drive to replay. find your drives at connect.comma.ai");
  cmd_parser.addOption({"demo", "use a demo route instead of providing your own"});
  cmd_parser.addOption({"auto", "Auto load the route from the best available source (no video): internal, openpilotci, comma_api, car_segments, testing_closet"});
  cmd_parser.addOption({"qcam", "load qcamera"});
  cmd_parser.addOption({"ecam", "load wide road camera"});
  cmd_parser.addOption({"dcam", "load driver camera"});
  cmd_parser.addOption({"msgq", "read can messages from the msgq"});
  cmd_parser.addOption({"panda", "read can messages from panda"});
  cmd_parser.addOption({"panda-serial", "read can messages from panda with given serial", "panda-serial"});
  if (SocketCanStream::available()) {
    cmd_parser.addOption({"socketcan", "read can messages from given SocketCAN device", "socketcan"});
  }
  cmd_parser.addOption({"zmq", "read can messages from zmq at the specified ip-address", "ip-address"});
  cmd_parser.addOption({"data_dir", "local directory with routes", "data_dir"});
  cmd_parser.addOption({"no-vipc", "do not output video"});
  cmd_parser.addOption({"dbc", "dbc file to open", "dbc"});
  cmd_parser.process(app);

  AbstractStream *stream = nullptr;

  if (cmd_parser.isSet("msgq")) {
    stream = new DeviceStream(&app);
  } else if (cmd_parser.isSet("zmq")) {
    stream = new DeviceStream(&app, cmd_parser.value("zmq"));
  } else if (cmd_parser.isSet("panda") || cmd_parser.isSet("panda-serial")) {
    try {
      stream = new PandaStream(&app, {.serial = cmd_parser.value("panda-serial")});
    } catch (std::exception &e) {
      qWarning() << e.what();
      return 0;
    }
  } else if (SocketCanStream::available() && cmd_parser.isSet("socketcan")) {
    stream = new SocketCanStream(&app, {.device = cmd_parser.value("socketcan")});
  } else {
    uint32_t replay_flags = REPLAY_FLAG_NONE;
    if (cmd_parser.isSet("ecam")) replay_flags |= REPLAY_FLAG_ECAM;
    if (cmd_parser.isSet("qcam")) replay_flags |= REPLAY_FLAG_QCAMERA;
    if (cmd_parser.isSet("dcam")) replay_flags |= REPLAY_FLAG_DCAM;
    if (cmd_parser.isSet("no-vipc")) replay_flags |= REPLAY_FLAG_NO_VIPC;

    const QStringList args = cmd_parser.positionalArguments();
    QString route;
    if (args.size() > 0) {
      route = args.first();
    } else if (cmd_parser.isSet("demo")) {
      route = DEMO_ROUTE;
    }
    if (!route.isEmpty()) {
      auto replay_stream = std::make_unique<ReplayStream>(&app);
      bool auto_source = cmd_parser.isSet("auto");
      if (!replay_stream->loadRoute(route, cmd_parser.value("data_dir"), replay_flags, auto_source)) {
        return 0;
      }
      stream = replay_stream.release();
    }
  }

  MainWindow w(stream, cmd_parser.value("dbc"));
  return app.exec();
}
