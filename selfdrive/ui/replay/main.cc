#include "selfdrive/ui/replay/replay.h"

#include <csignal>
#include <iostream>
#include <termios.h>

#include <QApplication>
#include <QCommandLineParser>
#include <QDebug>
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
  parser.addOption({"dcam", "load driver camera"});
  parser.addOption({"ecam", "load wide road camera"});
  parser.addOption({"qlog", "fallback to qlog without confirmation"});

  parser.process(app);
  const QStringList args = parser.positionalArguments();
  if (args.empty() && !parser.isSet("demo")) {
    parser.showHelp();
  }

  const QString route = args.empty() ? DEMO_ROUTE : args.first();
  QStringList allow = parser.value("allow").isEmpty() ? QStringList{} : parser.value("allow").split(",");
  QStringList block = parser.value("block").isEmpty() ? QStringList{} : parser.value("block").split(",");

  std::pair<QString, Replay::Flags> flag_map[] = {
    {"dcam", Replay::Flags::LoadDriverCam},
    {"ecam", Replay::Flags::LoadWideRoadCam},
    {"qlog", Replay::Flags::FallbackToQLog},
  };
  uint32_t flags = Replay::Flags::None;
  for (const auto &[key, flag] : flag_map) {
    if (parser.isSet(key)) {
      flags |= flag;
    }
  }
  Replay *replay = new Replay(route, allow, block, nullptr, flags, &app);
  if (!replay->load()) {
    app.exit(0);
    return 0;
  }

  replay->start(parser.value("start").toInt());
  // start keyboard control thread
  QThread *t = QThread::create(keyboardThread, replay);
  QObject::connect(t, &QThread::finished, t, &QThread::deleteLater);
  t->start();

  return app.exec();
}
