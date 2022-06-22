#include <sys/resource.h>

#include <QApplication>
#include <QTranslator>

#include "system/hardware/hw.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/window.h"

int main(int argc, char *argv[]) {
  setpriority(PRIO_PROCESS, 0, -20);

  qInstallMessageHandler(swagLogMessageHandler);
  initApp(argc, argv);

  QApplication a(argc, argv);

  QTranslator translator;
  if (!translator.load("main_fr", "/home/batman/openpilot/selfdrive/ui/translations")) {
    qDebug() << "Failed to load translation fr!";
  }
  a.installTranslator(&translator);  // needs to be before setting main window
//  QTranslator translator2;
//  if (!translator.load("main_es", "/home/batman/openpilot/selfdrive/ui/translations")) {
//    qDebug() << "Failed to load translation es!";
//  }
//  a.installTranslator(&translator2);

  MainWindow w;
  setMainWindow(&w);
  a.installEventFilter(&w);
  return a.exec();
}
