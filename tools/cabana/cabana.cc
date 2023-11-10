#include <QApplication>
#include <QCommandLineParser>

#include "selfdrive/ui/qt/util.h"
#include "tools/cabana/mainwin.h"
#include "tools/cabana/streamselector.h"
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
  cmd_parser.addOption({"qcam", "load qcamera"});
  cmd_parser.addOption({"ecam", "load wide road camera"});
  cmd_parser.addOption({"dcam", "load driver camera"});
  cmd_parser.addOption({"stream", "read can messages from live streaming"});
  cmd_parser.addOption({"panda", "read can messages from panda"});
  cmd_parser.addOption({"panda-serial", "read can messages from panda with given serial", "panda-serial"});
  if (SocketCanStream::available()) {
    cmd_parser.addOption({"socketcan", "read can messages from given SocketCAN device", "socketcan"});
  }
  cmd_parser.addOption({"zmq", "the ip address on which to receive zmq messages", "zmq"});
  cmd_parser.addOption({"data_dir", "local directory with routes", "data_dir"});
  cmd_parser.addOption({"no-vipc", "do not output video"});
  cmd_parser.addOption({"dbc", "dbc file to open", "dbc"});
  cmd_parser.process(app);

  QString dbc_file = cmd_parser.isSet("dbc") ? cmd_parser.value("dbc") : "";

  AbstractStream *stream = nullptr;
  if (cmd_parser.isSet("stream")) {
    stream = new DeviceStream(&app, cmd_parser.value("zmq"));
  } else if (cmd_parser.isSet("panda") || cmd_parser.isSet("panda-serial")) {
    PandaStreamConfig config = {};
    if (cmd_parser.isSet("panda-serial")) {
      config.serial = cmd_parser.value("panda-serial");
    }
    try {
      stream = new PandaStream(&app, config);
    } catch (std::exception &e) {
      qWarning() << e.what();
      return 0;
    }
  } else if (cmd_parser.isSet("socketcan")) {
    SocketCanStreamConfig config = {};
    config.device = cmd_parser.value("socketcan");
    stream = new SocketCanStream(&app, config);
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
      auto replay_stream = new ReplayStream(&app);
      if (!replay_stream->loadRoute(route, cmd_parser.value("data_dir"), replay_flags)) {
        return 0;
      }
      stream = replay_stream;
    }
  }

  int ret = 0;
  {
    MainWindow w;
    QTimer::singleShot(0, [&]() {
      if (!stream) {
        StreamSelector dlg(&stream);
        dlg.exec();
        dbc_file = dlg.dbcFile();
      }
      if (!stream) {
        stream = new DummyStream(&app);
      }
      stream->start();
      if (!dbc_file.isEmpty()) {
        w.loadFile(dbc_file);
      }
      w.show();
    });

    ret = app.exec();
  }

  delete can;
  return ret;
}
