#include <sys/resource.h>

#include <QApplication>
#include <QTranslator>
#include <QImage>
#include <QTimer>
#include <QPainter>
#include <QBuffer>
#include <QDebug>
#include <QPixmap>
#include <QScreen>
#include <QEventLoop>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include <QScopedPointer>

#include "system/hardware/hw.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/window.h"

int main(int argc, char *argv[]) {
  setpriority(PRIO_PROCESS, 0, -20);

  qInstallMessageHandler(swagLogMessageHandler);
  initApp(argc, argv);

  QApplication a(argc, argv);
  QCommandLineParser parser;
  parser.setApplicationDescription("OpenPilot UI with screen recording");
  parser.addHelpOption();
  parser.addVersionOption();

  QCommandLineOption outputOption(QStringList() << "o" << "output",
      "Output file path for the recorded video", "file");
  parser.addOption(outputOption);

  parser.process(a);

  if (!parser.isSet(outputOption)) {
    qCritical() << "Error: Output file path is required. Use --output or -o to specify it.";
    return -1;
  }

  QString outputFile = parser.value(outputOption);

  QTranslator translator;
  QString translation_file = QString::fromStdString(Params().get("LanguageSetting"));
  if (!translator.load(QString(":/%1").arg(translation_file)) && translation_file.length()) {
    qCritical() << "Failed to load translation file:" << translation_file;
  }

  a.installTranslator(&translator);

  MainWindow w;
  w.setAttribute(Qt::WA_DontShowOnScreen);
  w.setAttribute(Qt::WA_Mapped);
  w.setAttribute(Qt::WA_NoSystemBackground);
  w.resize(DEVICE_SCREEN_SIZE);
  setMainWindow(&w);
  a.installEventFilter(&w);

  return a.exec();
}
