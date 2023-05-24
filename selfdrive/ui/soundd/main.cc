#include <sys/resource.h>

#include <QApplication>

#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/soundd/sound.h"

int main(int argc, char **argv) {
  qInstallMessageHandler(swagLogMessageHandler);
  setpriority(PRIO_PROCESS, 0, -20);

  QApplication a(argc, argv);
  std::signal(SIGINT, sigTermHandler);
  std::signal(SIGTERM, sigTermHandler);

  Sound sound;
  return a.exec();
}
