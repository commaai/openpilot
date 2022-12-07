#include <QApplication>
#include <QDir>
#include <QCommandLineParser>
#include <QUuid>

#include "selfdrive/ui/qt/util.h"
#include "tools/cabana/mainwin.h"

int main(int argc, char *argv[]) {
  initApp(argc, argv);
  QApplication app(argc, argv);

  QCommandLineParser cmd_parser;
  cmd_parser.addHelpOption();
  cmd_parser.addPositionalArgument("route", "the drive to replay. find your drives at connect.comma.ai");
  cmd_parser.addOption({"demo", "use a demo route instead of providing your own"});
  cmd_parser.addOption({"qcam", "load qcamera"});
  cmd_parser.addOption({"data_dir", "local directory with routes", "data_dir"});
  cmd_parser.process(app);
  const QStringList args = cmd_parser.positionalArguments();
  if (args.empty() && !cmd_parser.isSet("demo")) {
    cmd_parser.showHelp();
  }

  QString uuid =  QUuid::createUuid().toString(QUuid::WithoutBraces);
  QString msgq_path = "/dev/shm/" + uuid;

  QDir dir;
  dir.mkdir(msgq_path);
  setenv("OPENPILOT_PREFIX", qPrintable(uuid), 1);

  int ret = 0;
  const QString route = args.empty() ? DEMO_ROUTE : args.first();
  CANMessages p(&app);
  if (p.loadRoute(route, cmd_parser.value("data_dir"), cmd_parser.isSet("qcam"))) {
    MainWindow w;
    w.showMaximized();
    ret = app.exec();
  }

  dir.rmdir(msgq_path);
  return ret;
}
