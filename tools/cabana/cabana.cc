#include <QApplication>
#include <QCommandLineParser>

#include "selfdrive/ui/qt/util.h"
#include "tools/cabana/mainwin.h"

const QString DEMO_ROUTE = "4cf7a6ad03080c90|2021-09-29--13-46-36";

int main(int argc, char *argv[]) {
  initApp(argc, argv);
  QApplication app(argc, argv);

  QCommandLineParser cmd_parser;
  cmd_parser.addHelpOption();
  cmd_parser.addPositionalArgument("route", "the drive to replay. find your drives at connect.comma.ai");
  cmd_parser.addOption({"demo", "use a demo route instead of providing your own"});
  cmd_parser.addOption({"data_dir", "local directory with routes", "data_dir"});
  cmd_parser.process(app);
  const QStringList args = cmd_parser.positionalArguments();
  if (args.empty() && !cmd_parser.isSet("demo")) {
    cmd_parser.showHelp();
  }

  const QString route = args.empty() ? DEMO_ROUTE : args.first();
  Parser p(&app);
  if (!p.loadRoute(route, cmd_parser.value("data_dir"), true)) {
    return 0;
  }
  MainWindow w;
  w.showMaximized();
  return app.exec();
}
