#include <termios.h>

#include <QApplication>
#include <QCommandLineParser>
#include <QDebug>
#include <QThread>
#include <csignal>
#include <iostream>

#include "selfdrive/ui/replay/replay.h"

const QString DEMO_ROUTE = "4cf7a6ad03080c90|2021-09-29--13-46-36";
struct termios oldt = {};
Replay *replay = nullptr;

void sigHandler(int s) {
  std::signal(s, SIG_DFL);
  if (oldt.c_lflag) {
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  }
  if (replay) {
    replay->stop();
  }
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

void keyboardThread(Replay *replay_) {
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
          replay_->seekTo(std::stoi(r) * 60, false);
        } else {
          replay_->seekTo(std::stoi(r), false);
        }
      } catch (std::invalid_argument) {
        qDebug() << "invalid argument";
      }
      getch();  // remove \n from entering seek
    } else if (c == 'm') {
      replay_->seekTo(+60, true);
    } else if (c == 'M') {
      replay_->seekTo(-60, true);
    } else if (c == 's') {
      replay_->seekTo(+10, true);
    } else if (c == 'S') {
      replay_->seekTo(-10, true);
    } else if (c == 'G') {
      replay_->seekTo(0, true);
    } else if (c == ' ') {
      replay_->pause(!replay_->isPaused());
    }
  }
}

void replayMessageOutput(QtMsgType type, const QMessageLogContext &context, const QString &msg) {
  QByteArray localMsg = msg.toLocal8Bit();
  if (type == QtDebugMsg) {
    std::cout << "\033[38;5;248m" << localMsg.constData() << "\033[00m" << std::endl;
  } else if (type == QtWarningMsg) {
    std::cout << "\033[38;5;227m" << localMsg.constData() << "\033[00m" << std::endl;
  } else if (type == QtCriticalMsg) {
    std::cout << "\033[38;5;196m" << localMsg.constData() << "\033[00m" << std::endl;
  } else {
    std::cout << localMsg.constData() << std::endl;
  }
}

int main(int argc, char *argv[]) {
  qInstallMessageHandler(replayMessageOutput);

  QApplication app(argc, argv);
  std::signal(SIGINT, sigHandler);
  std::signal(SIGTERM, sigHandler);

  const std::tuple<QString, REPLAY_FLAGS, QString> flags[] = {
      {"dcam", REPLAY_FLAG_DCAM, "load driver camera"},
      {"ecam", REPLAY_FLAG_ECAM, "load wide road camera"},
      {"no-loop", REPLAY_FLAG_NO_LOOP, "stop at the end of the route"},
      {"no-cache", REPLAY_FLAG_NO_FILE_CACHE, "turn off local cache"},
  };

  QCommandLineParser parser;
  parser.setApplicationDescription("Mock openpilot components by publishing logged messages.");
  parser.addHelpOption();
  parser.addPositionalArgument("route", "the drive to replay. find your drives at connect.comma.ai");
  parser.addOption({{"a", "allow"}, "whitelist of services to send", "allow"});
  parser.addOption({{"b", "block"}, "blacklist of services to send", "block"});
  parser.addOption({{"s", "start"}, "start from <seconds>", "seconds"});
  parser.addOption({"demo", "use a demo route instead of providing your own"});
  parser.addOption({"data_dir", "local directory with routes", "data_dir"});
  for (auto &[name, _, desc] : flags) {
    parser.addOption({name, desc});
  }

  parser.process(app);
  const QStringList args = parser.positionalArguments();
  if (args.empty() && !parser.isSet("demo")) {
    parser.showHelp();
  }

  const QString route = args.empty() ? DEMO_ROUTE : args.first();
  QStringList allow = parser.value("allow").isEmpty() ? QStringList{} : parser.value("allow").split(",");
  QStringList block = parser.value("block").isEmpty() ? QStringList{} : parser.value("block").split(",");

  uint32_t replay_flags = REPLAY_FLAG_NONE;
  for (const auto &[name, flag, _] : flags) {
    if (parser.isSet(name)) {
      replay_flags |= flag;
    }
  }
  replay = new Replay(route, allow, block, nullptr, replay_flags, parser.value("data_dir"), &app);
  if (!replay->load()) {
    return 0;
  }
  replay->start(parser.value("start").toInt());
  // start keyboard control thread
  QThread *t = new QThread();
  QObject::connect(t, &QThread::started, [=]() { keyboardThread(replay); });
  QObject::connect(t, &QThread::finished, t, &QThread::deleteLater);
  t->start();

  return app.exec();
}
