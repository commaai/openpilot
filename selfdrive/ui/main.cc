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

  QString language_file = QString::fromStdString(Params().get("LanguageSetting"));
  qDebug() << "Loading language:" << language_file;

  QTranslator translator;
  if (!translator.load(language_file, "translations")) {
    qDebug() << "Failed to load translation file!";
  }
  QApplication a(argc, argv);
  a.installTranslator(&translator);

  qDebug() << "Before MainWindow";
  MainWindow w;
  qDebug() << "Before MainWindow";
  setMainWindow(&w);
  qDebug() << "Before MainWindow";
  a.installEventFilter(&w);
  qDebug() << "Before MainWindow";
  return a.exec();
}
