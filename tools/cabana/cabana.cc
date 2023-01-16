#include <QApplication>
#include <QCommandLineParser>

#include "common/prefix.h"
#include "selfdrive/ui/qt/util.h"
#include "tools/cabana/mainwin.h"

int main(int argc, char *argv[]) {
  QCoreApplication::setApplicationName("Cabana");
  QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
  initApp(argc, argv);
  QApplication app(argc, argv);

  QCommandLineParser cmd_parser;
  cmd_parser.addHelpOption();
  cmd_parser.addPositionalArgument("route", "the drive to replay. find your drives at connect.comma.ai");
  cmd_parser.addOption({"demo", "use a demo route instead of providing your own"});
  cmd_parser.addOption({"qcam", "load qcamera"});
  cmd_parser.addOption({"ecam", "load wide road camera"});
  cmd_parser.addOption({"data_dir", "local directory with routes", "data_dir"});
  cmd_parser.process(app);
  const QStringList args = cmd_parser.positionalArguments();
  if (args.empty() && !cmd_parser.isSet("demo")) {
    cmd_parser.showHelp();
  }


  const QString route = args.empty() ? DEMO_ROUTE : args.first();
  uint32_t replay_flags = REPLAY_FLAG_NONE;
  if (cmd_parser.isSet("ecam")) {
    replay_flags |= REPLAY_FLAG_ECAM;
  } else if (cmd_parser.isSet("qcam")) {
    replay_flags |= REPLAY_FLAG_QCAMERA;
  }

  // TODO: Remove when OpenpilotPrefix supports ZMQ
#ifndef __APPLE__
  OpenpilotPrefix op_prefix;
#endif

  CANMessages p(&app);
  int ret = 0;
  if (p.loadRoute(route, cmd_parser.value("data_dir"), replay_flags)) {
    MainWindow w;
    w.show();
    ret = app.exec();
  }
  return ret;
}
