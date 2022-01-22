#include <QApplication>
#include <QCommandLineParser>
#include <QDebug>
#include <QThread>

#include "selfdrive/ui/replay/replay.h"
#include "selfdrive/ui/replay/consoleui.h"

#include <ncurses.h>

const QString DEMO_ROUTE = "4cf7a6ad03080c90|2021-09-29--13-46-36";
Replay *replay = nullptr;


WINDOW *window = nullptr;
void replayMessageOutput(QtMsgType type, const QMessageLogContext &context, const QString &msg) {
  if (!window) return;

  QByteArray localMsg = msg.toLocal8Bit();
  wprintw(window, "%s\n", localMsg.constData());
  wrefresh(window);
  // if (type == QtDebugMsg) {
  //   std::cout << "\033[38;5;248m" << localMsg.constData() << "\033[00m" << std::endl;
  // } else if (type == QtWarningMsg) {
  //   std::cout << "\033[38;5;227m" << localMsg.constData() << "\033[00m" << std::endl;
  // } else if (type == QtCriticalMsg) {
  //   std::cout << "\033[38;5;196m" << localMsg.constData() << "\033[00m" << std::endl;
  // } else {
  //   std::cout << localMsg.constData() << std::endl;
  // }
}

int main(int argc, char *argv[]) {
  qInstallMessageHandler(replayMessageOutput);
  QApplication app(argc, argv);

  const std::tuple<QString, REPLAY_FLAGS, QString> flags[] = {
      {"dcam", REPLAY_FLAG_DCAM, "load driver camera"},
      {"ecam", REPLAY_FLAG_ECAM, "load wide road camera"},
      {"no-loop", REPLAY_FLAG_NO_LOOP, "stop at the end of the route"},
      {"no-cache", REPLAY_FLAG_NO_FILE_CACHE, "turn off local cache"},
      {"qcam", REPLAY_FLAG_QCAMERA, "load qcamera"},
      {"yuv", REPLAY_FLAG_SEND_YUV, "send yuv frame"},
      {"no-cuda", REPLAY_FLAG_NO_CUDA, "disable CUDA"},
      {"no-vipc", REPLAY_FLAG_NO_VIPC, "do not output video"},
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

  system("clear");
  auto win = initscr();
  clear();
  cbreak();
  noecho();
  printw("Route %s\n", qPrintable(route));
  int height, width;
  getmaxyx(stdscr, height, width);
  window = newwin(height - 2, width - 2, 5, 1);
  scrollok(window, true);
  refresh();
  keypad(win, true);
  nodelay(win, true);  // getch() is a non-blocking call

  Keyboard keyboard(replay);
  int ret = app.exec();
  clrtoeol();
  refresh();
  endwin();
  return ret;
}
