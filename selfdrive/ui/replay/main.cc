#include "selfdrive/ui/replay/replay.h"

#include <csignal>
#include <iostream>
#include <termios.h>

#include <QApplication>
#include <QCommandLineParser>
#include <QDebug>
#include <QThread>

const QString DEMO_ROUTE = "4cf7a6ad03080c90|2021-09-29--13-46-36";

void sigHandler(int s) {
  std::signal(s, SIG_DFL);
  qApp->quit();
}

int getch() {
  int ch;
  struct termios oldt;
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
          replay->seekTo(std::stoi(r) * 60);
        } else {
          replay->seekTo(std::stoi(r));
        }
      } catch (std::invalid_argument) {
        qDebug() << "invalid argument";
      }
      getch();  // remove \n from entering seek
    } else if (c == 'm') {
      replay->relativeSeek(+60);
    } else if (c == 'M') {
      replay->relativeSeek(-60);
    } else if (c == 's') {
      replay->relativeSeek(+10);
    } else if (c == 'S') {
      replay->relativeSeek(-10);
    } else if (c == 'G') {
      replay->relativeSeek(0);
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

  parser.process(app);
  const QStringList args = parser.positionalArguments();
  if (args.empty() && !parser.isSet("demo")) {
    parser.showHelp();
  }

  const QString route = args.empty() ? DEMO_ROUTE : args.first();
  QStringList allow = parser.value("allow").isEmpty() ? QStringList{} : parser.value("allow").split(",");
  QStringList block = parser.value("block").isEmpty() ? QStringList{} : parser.value("block").split(",");

  Replay *replay = new Replay(route, allow, block, nullptr, parser.isSet("dcam"), parser.isSet("ecam"), &app);
  if (replay->load()) {
    replay->start(parser.value("start").toInt());
  }

  // start keyboard control thread
  QThread *t = QThread::create(keyboardThread, replay);
  QObject::connect(t, &QThread::finished, t, &QThread::deleteLater);
  t->start();

  return app.exec();
}
