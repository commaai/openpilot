#include <sys/resource.h>
#include <unistd.h>
#include <libgen.h>

#include <QApplication>
#include <QTranslator>

#include "system/hardware/hw.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/window.h"

int main(int argc, char *argv[]) {
  setpriority(PRIO_PROCESS, 0, -20);

  // Set working directory so asset paths are valid independent of launch path
  char path[PATH_MAX];
  memset(path, 0, sizeof(path));
  if (readlink("/proc/self/exe", path, sizeof(path)) == -1) {
    perror("readlink failed");
    return EXIT_FAILURE;
  }
  if (chdir(dirname(path)) == -1) {
    perror("chdir failed");
    return EXIT_FAILURE;
  }

  qInstallMessageHandler(swagLogMessageHandler);
  initApp(argc, argv);

  QTranslator translator;
  QString translation_file = QString::fromStdString(Params().get("LanguageSetting"));
  if (!translator.load(QString(":/%1").arg(translation_file)) && translation_file.length()) {
    qCritical() << "Failed to load translation file:" << translation_file;
  }

  QApplication a(argc, argv);
  a.installTranslator(&translator);

  MainWindow w;
  setMainWindow(&w);
  a.installEventFilter(&w);
  return a.exec();
}
