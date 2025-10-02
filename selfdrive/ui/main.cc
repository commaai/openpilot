#include <sys/resource.h>

#include <QApplication>
#include <QTranslator>
#include <QScreen>
#include <QDebug>

#include "system/hardware/hw.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/window.h"

int main(int argc, char *argv[]) {
  setpriority(PRIO_PROCESS, 0, -20);

  qInstallMessageHandler(swagLogMessageHandler);
  initApp(argc, argv);

  QTranslator translator;
  QString translation_file = QString::fromStdString(Params().get("LanguageSetting"));
  if (!translator.load(QString(":/%1").arg(translation_file)) && translation_file.length()) {
    qCritical() << "Failed to load translation file:" << translation_file;
  }

  QApplication a(argc, argv);
  a.installTranslator(&translator);

  // Debug DPI/scaling info
  const char *sf = qgetenv("QT_SCALE_FACTOR").constData();
  auto scr = a.primaryScreen();
  qDebug() << "QT_SCALE_FACTOR=" << (sf ? sf : "")
           << " devicePixelRatio=" << (scr ? scr->devicePixelRatio() : 0)
           << " devicePixelRatioF=" << (scr ? scr->devicePixelRatio() : 0)
           << " logicalDpi=" << (scr ? scr->logicalDotsPerInch() : 0)
           << " physicalDpi=" << (scr ? scr->physicalDotsPerInch() : 0);

  MainWindow w;
  setMainWindow(&w);
  a.installEventFilter(&w);
  return a.exec();
}
