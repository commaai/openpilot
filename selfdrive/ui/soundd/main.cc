#include <sys/resource.h>

#include <QApplication>

#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/soundd/sound.h"

void sigHandler(int s) {
  qApp->quit();
}

int main(int argc, char **argv) {
  qInstallMessageHandler(swagLogMessageHandler);
  setpriority(PRIO_PROCESS, 0, -20);

  QApplication a(argc, argv);
  std::signal(SIGINT, sigHandler);
  std::signal(SIGTERM, sigHandler);

  Sound sound;
  return a.exec();
}
