#include <QApplication>
#include <QCommandLineParser>

#include "selfdrive/ui/qt/util.h"
#include "tools/cabana/mainwin.h"
#include "tools/replay/replay.h"

const QString DEMO_ROUTE = "4cf7a6ad03080c90|2021-09-29--13-46-36";

int main(int argc, char *argv[]) {
  initApp(argc, argv);
  QApplication app(argc, argv);
  QCommandLineParser parser;
  parser.addHelpOption();
  parser.addPositionalArgument("route", "the drive to replay. find your drives at connect.comma.ai");
  parser.addOption({"demo", "load driver camera"});
  parser.addOption({"data_dir", "local directory with routes", "data_dir"});
  parser.process(app);
  const QStringList args = parser.positionalArguments();
  if (args.empty() && !parser.isSet("demo")) {
    parser.showHelp();
  }

  const QString route = args.empty() ? DEMO_ROUTE : args.first();
  Replay *replay = new Replay(route, {"can", "roadEncodeIdx"}, {}, nullptr, 0, parser.value("data_dir"), &app);
  if (!replay->load()) {
    return 0;
  }

  replay->start(parser.value("start").toInt());
  MainWindow w;
  w.showMaximized();
  return app.exec();
}
