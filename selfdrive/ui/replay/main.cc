#include "selfdrive/ui/replay/replay.h"

#include <csignal>
#include <iostream>
#include <optional>
#include <termios.h>

#include <QApplication>
#include <QCommandLineParser>
#include <QDebug>
#include <QRegExp>
#include <QThread>

const QString DEMO_ROUTE = "4cf7a6ad03080c90|2021-09-29--13-46-36";

struct termios oldt = {};

void sigHandler(int s) {
  std::signal(s, SIG_DFL);
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  qApp->quit();
}

int getch() {
  int ch;
  struct termios newt;

  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);

  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  ch = getchar();
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);

  return ch;
}

void keyboardThread(Replay *replay) {
  char c;
  while (true) {
    c = getch();
    if (c == '\n') {
      printf("Enter seek request: ");
      std::string r;
      std::cin >> r;

      try {
        if (r[0] == '#') {
          r.erase(0, 1);
          replay->seekTo(std::stoi(r) * 60, false);
        } else {
          replay->seekTo(std::stoi(r), false);
        }
      } catch (std::invalid_argument) {
        qDebug() << "invalid argument";
      }
      getch();  // remove \n from entering seek
    } else if (c == 'm') {
      replay->seekTo(+60, true);
    } else if (c == 'M') {
      replay->seekTo(-60, true);
    } else if (c == 's') {
      replay->seekTo(+10, true);
    } else if (c == 'S') {
      replay->seekTo(-10, true);
    } else if (c == 'G') {
      replay->seekTo(0, true);
    } else if (c == ' ') {
      replay->pause(!replay->isPaused());
    }
  }
}

struct RouteIdentifier {
  QString dongle_id;
  QString timestamp;
  int segment_id;
  QString str;
};

bool parse_route(const QString &str, RouteIdentifier &route) {
  QRegExp rx(R"(^([a-z0-9]{16})([|_/])(\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})(?:(--|/)(\d*))?$)");
  if (rx.indexIn(str) == -1) {
    return false;
  }
  const QStringList list = rx.capturedTexts();
  route = {list[1], list[3], list[5].toInt(), str};
  return true;
}

int main(int argc, char *argv[]){
  QApplication app(argc, argv);
  std::signal(SIGINT, sigHandler);
  std::signal(SIGTERM, sigHandler);

  QCommandLineParser parser;
  parser.setApplicationDescription("Mock openpilot components by publishing logged messages.");
  parser.addHelpOption();
  parser.addPositionalArgument("route", "the drive to replay. find your drives at connect.comma.ai");
  parser.addOption({{"a", "allow"}, "whitelist of services to send", "allow"});
  parser.addOption({{"b", "block"}, "blacklist of services to send", "block"});
  parser.addOption({{"s", "start"}, "start from <seconds>", "seconds"});
  parser.addOption({"demo", "use a demo route instead of providing your own"});
  parser.addOption({"data_dir", "local directory with routes", "data_dir"});
  parser.addOption({"dcam", "load driver camera"});
  parser.addOption({"ecam", "load wide road camera"});

  parser.process(app);
  const QStringList args = parser.positionalArguments();
  if (args.empty() && !parser.isSet("demo")) {
    parser.showHelp();
  }

  QStringList allow = parser.value("allow").isEmpty() ? QStringList{} : parser.value("allow").split(",");
  QStringList block = parser.value("block").isEmpty() ? QStringList{} : parser.value("block").split(",");

  RouteIdentifier route;
  if (!parse_route(args.empty() ? DEMO_ROUTE : args.first(), route)) {
    qDebug() << "invalid route format";
    return 0;
  }
  qDebug() << route.dongle_id << route.timestamp << route.segment_id;
  Replay *replay = new Replay(route.str, allow, block, nullptr, parser.isSet("dcam"), parser.isSet("ecam"), parser.value("data_dir"), &app);
  if (!replay->load()) {
    return 0;
  }
  int start_from = route.segment_id * 60 + parser.value("start").toInt();
  replay->start(start_from);

  // start keyboard control thread
  QThread *t = QThread::create(keyboardThread, replay);
  QObject::connect(t, &QThread::finished, t, &QThread::deleteLater);
  t->start();

  return app.exec();
}
